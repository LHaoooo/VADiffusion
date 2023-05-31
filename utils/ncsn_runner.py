# writer ： Liuhao
# create_time ： 2023/5/13 13:30
# file_name：ncsn_runner.py

import datetime
import os
import time
import logging
import imageio
import math
import matplotlib
import torch.nn as nn
from loss.dsm import anneal_dsm_score_estimation
from utils.eval_util import save_evaluation_curves_pred
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import psutil
import scipy.stats as st
import sys
import time
import yaml

import torch
import torch.nn.functional as F
import torchvision.transforms as Transforms

from cv2 import rectangle, putText
from functools import partial
from math import ceil, log10
from multiprocessing import Process
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from torch.distributions.gamma import Gamma
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resized_crop
from torchvision.utils import make_grid, save_image
from loss import get_optimizer, warmup_lr
from models.MV_reconAE import reconAE

from models import (ddpm_sampler,
                    ddim_sampler,)
# from models.better.ncsnpp_more import UNetMore_DDPM
from dataset.YUVvideo_dataset import VideoDataset
from models.ema import EMAHelper

__all__ = ['NCSNRunner']

METADATA = {
    "UCSD_ped2": {
        "testing_video_num": 12,
        "testing_frames_cnt": [180, 180, 150, 180, 150, 180, 180, 180, 120, 150,
                               180, 180]
    },
    "avenue": {
        "testing_video_num": 21,
        "testing_frames_cnt": [1439, 1211, 923, 947, 1007, 1283, 605, 36, 1175, 841,
                               472, 1271, 549, 507, 1001, 740, 426, 294, 248, 273,
                               76],
    },
    "shanghaitech": {
        "testing_video_num": 107,
        "testing_frames_cnt": [265, 433, 337, 601, 505, 409, 457, 313, 409, 337,
                               337, 457, 577, 313, 529, 193, 289, 289, 265, 241,
                               337, 289, 265, 217, 433, 409, 529, 313, 217, 241,
                               313, 193, 265, 317, 457, 337, 361, 529, 409, 313,
                               385, 457, 481, 457, 433, 385, 241, 553, 937, 865,
                               505, 313, 361, 361, 529, 337, 433, 481, 649, 649,
                               409, 337, 769, 433, 241, 217, 265, 265, 217, 265,
                               409, 385, 481, 457, 313, 601, 241, 481, 313, 337,
                               457, 217, 241, 289, 337, 313, 337, 265, 265, 337,
                               361, 433, 241, 433, 601, 505, 337, 601, 265, 313,
                               241, 289, 361, 385, 217, 337, 265]
    },

}

def count_training_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_proc_mem():
    return psutil.Process(os.getpid()).memory_info().rss /1024**3

def get_GPU_mem():
    try:
        num = torch.cuda.device_count()
        mem = 0
        for i in range(num):
            mem_free, mem_total = torch.cuda.mem_get_info(i)
            mem += (mem_total - mem_free)/1024**3
        return mem
    except:
        return 0

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99, save_seq=True):
        self.momentum = momentum
        self.save_seq = save_seq
        if self.save_seq:
            self.vals, self.steps = [], []
        self.reset()

    def reset(self):
        self.val, self.avg = None, 0

    def update(self, val, step=None):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        if self.save_seq:
            self.vals.append(val)
            if step is not None:
                self.steps.append(step)
    
def get_model(config):
    arch = getattr(config.model, 'arch', 'unetmore')  # unetmore unetmore3D unetmorepseudo3d

    if arch in ['unetmore', 'unetmore3d', 'unetmorepseudo3d']:
        from models.better.ncsnpp_more import UNetMore_DDPM # This lets the code run on CPU when 'unetmore' is not used
        return UNetMore_DDPM(config).to(config.device)
    else:
        Exception("arch is not valid [unetmore, unetmore3d, unetmorepseudo3d]")

def conditioning_fn(config, X, num_frames_pred=1, prob_mask_cond=0.0, prob_mask_future=0.0, conditional=True):
    imsize = config.data.image_size
    if not conditional:
        return X.reshape(len(X), -1, imsize, imsize), None, None

    pred = num_frames_pred
    # Frames to train on / sample
    target_frames = X[:, -config.model.ImgChnNum*pred:,:,:]

    # Condition (Past)
    cond_frames = X[:, :-config.model.ImgChnNum*pred,:,:]

    if prob_mask_cond > 0.0:
        cond_mask = (torch.rand(X.shape[0], device=X.device) > prob_mask_cond)
        cond_frames = cond_mask.reshape(-1, 1, 1, 1) * cond_frames
        cond_mask = cond_mask.to(torch.int32) # make 0,1
    else:
        cond_mask = None

    return target_frames, cond_frames, cond_mask   # , future_mask


class NCSNRunner():
    def __init__(self, args, config, 
                 trainset_yuvroot,testset_yuvroot,trainset_mvroot,testset_mvroot):
        self.args = args
        self.config = config
        # self.config_uncond = config_uncond
        self.trainset_yuvroot, self.testset_yuvroot = trainset_yuvroot, testset_yuvroot
        self.trainset_mvroot, self.testset_mvroot = trainset_mvroot, testset_mvroot

        self.version = getattr(self.config.model, 'version', "DDPM")
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

        # condition 中过去帧的数量和mask概率
        self.condf, self.condp = self.config.data.num_frames_cond, getattr(self.config.data, "prob_mask_cond", 0.0)
        if self.condp == 0.0 or self.condp > 0.0:
            self.mode_pred = "one"  # prediction
        
        self.step_size=16
        self.test_interval=4

    def get_time(self):
        curr_time = time.time()
        curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
        elapsed = str(datetime.timedelta(seconds=(curr_time - self.start_time)))
        return curr_time_str, elapsed

    def convert_time_stamp_to_hrs(self, time_day_hr):
        time_day_hr = time_day_hr.split(",")
        if len(time_day_hr) > 1:
            days = time_day_hr[0].split(" ")[0]
            time_hr = time_day_hr[1]
        else:
            days = 0
            time_hr = time_day_hr[0]
        # Hr
        hrs = time_hr.split(":")
        return float(days)*24 + float(hrs[0]) + float(hrs[1])/60 + float(hrs[2])/3600

    def get_sampler(self):
        # Sampler
        if self.version == "DDPM":
            sampler = partial(ddpm_sampler, config=self.config)
        elif self.version == "DDIM":
            sampler = partial(ddim_sampler, config=self.config)

        return sampler
    
    def train(self):

        dataset = VideoDataset(self.config.model.ImgChnNum, self.config.model.sampled_mv_num,
                                self.trainset_yuvroot, self.trainset_mvroot)  # 此时并未设置last_mv = true,所以输出的是4个gop的MV
        dataloader = DataLoader(dataset=dataset, batch_size=self.config.training.batch_size, 
                                num_workers=self.config.data.num_workers, shuffle=True)
        
        test_dataset = VideoDataset(self.config.model.ImgChnNum, self.config.model.sampled_mv_num,
                                     self.testset_yuvroot, self.testset_mvroot)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.config.eval.batch_size, 
                                 num_workers=self.config.data.num_workers, shuffle=False)

        scorenet = get_model(self.config)  # UNetMore_DDPM
        scorenet = torch.nn.DataParallel(scorenet)

        logging.info(f"Number of parameters: {count_parameters(scorenet)}")
        logging.info(f"Number of trainable parameters: {count_training_parameters(scorenet)}")

        # MV recon net
        recon_MV_dict = torch.load(self.config.model.recon_MV_pretrained)["model_state_dict"]
        recon_AE = reconAE(num_in_ch=self.config.model.motion_channels*self.config.model.sampled_mv_num, 
                            seq_len=1, features_root=self.config.model.feature_root,
                            skip_ops=self.config.model.skip_ops).to(self.config.device)
        recon_AE = torch.nn.DataParallel(recon_AE)
        recon_AE.load_state_dict(recon_MV_dict, False)
        for param in recon_AE.parameters():
            param.requires_grad = False
        recon_AE.eval()

        optimizer = get_optimizer(self.config, scorenet.parameters())
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.8)

        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            logging.info(f"Number of GPUs : {num_devices}")
            for i in range(num_devices):
                logging.info(torch.cuda.get_device_properties(i))
        else:
            logging.info(f"Running on CPU!")

        start_epoch = 0
        step = 0

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(scorenet)

        if self.args.resume_training:
            assert(self.config.model.pretrained is not None)
            states = torch.load(self.config.model.pretrained)
            scorenet.load_state_dict(states[0])
            ### Make sure we can resume with different eps
            states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
            logging.info(f"Resuming training from checkpoint.pt in {self.config.model.pretrained} at epoch {start_epoch}, step {step}.")

        # print(scorenet)
        net = scorenet.module if hasattr(scorenet, 'module') else scorenet

        # Conditional
        conditional = self.config.model.clip_hist > 0
        cond, test_cond = None, None

        # Initialize meters
        self.init_meters()

        # Sampler
        sampler = self.get_sampler()

        self.total_train_time = 0
        self.start_time = time.time()

        early_end = False
        best_auc = -1
        val_times = 0
        txtpath=os.path.join(self.args.log_path, 'auc.txt')##the file of save auc
        if (os.path.exists(txtpath)) :
            os.remove(txtpath)
        
        for epoch in range(start_epoch, self.config.training.n_epochs):
            for batch, train_data in tqdm(enumerate(dataloader),desc="Training Epoch %d" % (epoch + 1),total=len(dataloader)):

                optimizer.zero_grad()
                lr = warmup_lr(optimizer, step, getattr(self.config.optim, 'warmup', 0), self.config.optim.lr) # lr预热，等模型稳定了再使用初始lr
                scorenet.train()
                step += 1

                # Data
                sample_frames, sample_mvs, _, _, mv_last = train_data  # 取出一个batch的frames
                # print(sample_frames.shape)  # 8,5,256,256
                # print(sample_mvs.shape)  # 8,24,256,256
                # print(mv_last.shape)  # 8,64,64,2
                sample_frames = sample_frames.to(self.config.device)  # 前4个I作为cond，第五个作为predtarget
                
                # target I frame , condition frames, mask prob
                sample_frames, cond, cond_mask = conditioning_fn(self.config, sample_frames, num_frames_pred=self.config.model.clip_pred,
                                                     prob_mask_cond=getattr(self.config.data, 'prob_mask_cond', 0.0),
                                                     prob_mask_future=getattr(self.config.data, 'prob_mask_future', 0.0),
                                                     conditional=conditional)
                
                sample_mvs = sample_mvs.to(self.config.device)

                # MV recon
                mv_recon = torch.zeros_like(sample_mvs)
                y_ch = self.config.model.motion_channels*self.config.model.sampled_mv_num
                for j in range(self.config.model.clip_hist):
                    _, reconAE_out = recon_AE(sample_mvs[:, y_ch * j:y_ch * (j + 1), :, :])
                    mv_recon[:, y_ch * j:y_ch * (j + 1), :, :] = reconAE_out
                upsample=nn.Upsample(scale_factor=4, mode='nearest')
                sample_mvs = upsample(mv_recon)
                # sample_mvs = F.interpolate(mv_recon, size=(256, 256), mode='bilinear', align_corners=False)
                # print("samplemv's shape: ",sample_mvs.shape)  # 8,24,256,256
                sample_mvs_split = sample_mvs.view(sample_mvs.shape[0],4,6,256,256)

                cond_ip = np.zeros((sample_mvs.shape[0], 28,256,256))
                for i in range(4):
                    temp = sample_mvs_split[:,i,:,:,:]
                    # print("temp shape: ", temp.shape)  # 8,6,256,256
                    temp = temp.squeeze(1)  
                    # print("squeeze temp shape: ", temp.shape)  # 8,6,256,256
                    cond_ip[:,7*i:7*(i+1),:,:] = torch.cat((cond[:,i,:,:].unsqueeze(1).cpu(), temp.cpu()), dim =1)

                cond_ip = torch.from_numpy(cond_ip).to(self.config.device)
                
                # Loss
                itr_start = time.time()
                hook = None
                loss = anneal_dsm_score_estimation(scorenet, sample_frames, labels=None, cond=cond_ip, cond_mask=cond_mask,
                                                                    loss_type=getattr(self.config.training, 'loss_type', 'a'),
                                                                    gamma=getattr(self.config.model, 'gamma', False),
                                                                    L1=getattr(self.config.training, 'L1', False), hook=hook,
                                                                    all_frames=getattr(self.config.model, 'output_all_frames', False))

                # Optimize
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(scorenet.parameters(), getattr(self.config.optim, 'grad_clip', np.inf))  # 梯度裁剪，防止梯度爆炸
                optimizer.step()

                # Training time
                itr_time = time.time() - itr_start
                self.total_train_time += itr_time
                self.time_train.update(self.convert_time_stamp_to_hrs(str(datetime.timedelta(seconds=self.total_train_time))) + self.time_train_prev)

                # Record
                self.losses_train.update(loss.item(), step)
                self.epochs.update(epoch + (batch + 1)/len(dataloader))
                self.lr_meter.update(lr)
                self.grad_norm.update(grad_norm.item())
                if step == 1 or step % getattr(self.config.training, "log_freq", 1) == 0:
                    logging.info("elapsed: {}, train time: {:.04f}, mem: {:.03f}GB, GPUmem: {:.03f}GB, step: {}, lr: {:.06f}, grad: {:.04f}, loss: {:.04f}".format(
                        str(datetime.timedelta(seconds=(time.time() - self.start_time)) + datetime.timedelta(seconds=self.time_elapsed_prev*3600))[:-3],
                        self.time_train.val, get_proc_mem(), get_GPU_mem(), step, lr, grad_norm, loss.item()))

                if self.config.model.ema:
                    ema_helper.update(scorenet)

                if step >= self.config.training.n_iters:  # 后续加入早停条件
                    early_end = True
                    break

                # Save model
                if (step % 1000 == 0 and step != 0):
                    states = [
                        scorenet.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())
                    ckpt_path = os.path.join(self.args.log_path, 'checkpoint_{}.pth'.format(step))
                    logging.info(f"Saving checkpoint.pt in {self.args.log_path}")
                    torch.save(states, ckpt_path)

                # Plot graphs
                # try:
                #     plot_graphs_process.join()
                # except:
                #     pass
                # plot_graphs_process = Process(target=self.plot_graphs)
                # plot_graphs_process.start()
                
                test_scorenet = None
                # Get test_scorenet
                # if step == 1 or (step >= 10000 and step % self.config.training.val_freq == 0):
                if step >= 15000 and step % self.config.training.val_freq == 0 :

                    if self.config.model.ema:
                        test_scorenet = ema_helper.ema_copy(scorenet)
                    else:
                        test_scorenet = scorenet

                    test_scorenet.eval()
             
                # Validation
                # if step == 1 or (step >= 10000 and step % self.config.training.val_freq == 0):
                if step >= 15000 and step % self.config.training.val_freq == 0 :
                    dataset_name = self.config.data.dataset
                    
                    score_func = nn.MSELoss(reduction="none")
                    # bbox anomaly scores for each frame
                    video_list = [name for name in os.listdir(self.testset_yuvroot)]
                    video_list.sort()
                    #print(video_list)
                    frame_scores=[]
                    for k in range(len(video_list)):
                        m=[0 for i in range((METADATA[dataset_name]["testing_frames_cnt"])[k])]
                        frame_scores.append(m)

                    for test_data in tqdm(test_loader, desc="Eval:", total=len(test_loader)):
                        test_X, sample_mvs_t, pred_frame_test, v_name, mv_last = test_data
                        test_X = test_X.to(self.config.device)
                        test_X, test_cond, test_cond_mask = conditioning_fn(self.config, test_X, num_frames_pred=self.config.model.clip_pred,
                                                                            prob_mask_cond=getattr(self.config.data, 'prob_mask_cond', 0.0),
                                                                            prob_mask_future=getattr(self.config.data, 'prob_mask_future', 0.0),
                                                                            conditional=conditional)
                        
                        # MV recon
                        mv_recon_t = torch.zeros_like(sample_mvs_t)
                        y_ch = self.config.model.motion_channels*self.config.model.sampled_mv_num
                        
                        for j in range(self.config.model.clip_hist):
                            _, reconAE_out = recon_AE(sample_mvs_t[:, y_ch * j:y_ch * (j + 1), :, :])
                            mv_recon_t[:, y_ch * j:y_ch * (j + 1), :, :] = reconAE_out
                        upsample=nn.Upsample(scale_factor=4, mode='nearest')
                        sample_mvs_t1 = upsample(mv_recon_t)
                        # sample_mvs_t1 = F.interpolate(mv_recon_t, size=(256, 256), mode='bilinear', align_corners=False)
                        # print("samplemv's shape: ",sample_mvs_t.shape)
                        
                        sample_mvs_t_split = sample_mvs_t1.view(sample_mvs_t1.shape[0],4,6,256,256)
                        cond_ip_t = np.zeros((sample_mvs_t1.shape[0], 28,256,256))

                        # Initial samples
                        n_init_samples = sample_mvs_t1.shape[0]
                        channel_num = self.config.model.ImgChnNum*self.config.model.clip_pred
                        # channel_num = self.config.model.ImgChnNum + self.config.model.clip_hist*(self.config.model.sampled_mv_num*self.config.model.motion_channels+ self.config.model.ImgChnNum)
                        # init_samples_shape = (n_init_samples, self.config.data.channels*self.config.data.num_frames, self.config.data.image_size, self.config.data.image_size)
                        init_samples_shape = (n_init_samples, channel_num, self.config.data.image_size, self.config.data.image_size)
                        # print("init_sample_shape:", init_samples_shape)

                        if self.version == "DDPM" or self.version == "DDIM" :
                            if getattr(self.config.model, 'gamma', False):    # 使用gamma分布
                                used_k, used_theta = net.k_cum[0], net.theta_t[0]
                                z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
                                init_samples = z - used_k*used_theta # we don't scale here
                            else:
                                init_samples = torch.randn(init_samples_shape, device=self.config.device)  # 使用高斯分布随机产生
                        for i in range(4):
                            temp = sample_mvs_t_split[:,i,:,:,:]
                            # temp = temp.squeeze(1)
                            cond_ip_t[:,7*i:7*(i+1),:,:] = torch.cat((test_cond[:,i,:,:].unsqueeze(1).cpu(), temp.cpu()), dim =1)

                        cond_ip_t = torch.from_numpy(cond_ip_t).to(self.config.device)

                        with torch.no_grad():
                            test_hook = None
                            test_dsm_loss = anneal_dsm_score_estimation(test_scorenet, test_X, labels=None, cond=cond_ip_t, cond_mask=test_cond_mask,
                                                                        loss_type=getattr(self.config.training, 'loss_type', 'a'),
                                                                        gamma=getattr(self.config.model, 'gamma', False),
                                                                        L1=getattr(self.config.training, 'L1', False), hook=test_hook,
                                                                        all_frames=getattr(self.config.model, 'output_all_frames', False))
                        
                        self.losses_test.update(test_dsm_loss.item(), step)
                        logging.info("elapsed: {}, step: {}, mem: {:.03f}GB, GPUmem: {:.03f}GB, test_loss: {:.04f} ".format(
                            str(datetime.timedelta(seconds=(time.time() - self.start_time)) + datetime.timedelta(seconds=self.time_elapsed_prev*3600))[:-3],
                            step, get_proc_mem(), get_GPU_mem(), test_dsm_loss.item()))

                        with torch.no_grad():
                            # frame sample for prediction
                            all_samples = sampler(init_samples, test_scorenet, cond=cond_ip_t, cond_mask=test_cond_mask,
                                                n_steps_each=self.config.sampling.n_steps_each,
                                                step_lr=self.config.sampling.step_lr, just_beta=False,
                                                final_only=True, denoise=self.config.sampling.denoise,
                                                subsample_steps=getattr(self.config.sampling, 'subsample', None),
                                                clip_before=getattr(self.config.sampling, 'clip_before', True),
                                                verbose=False, log=False, gamma=getattr(self.config.model, 'gamma', False)).to('cpu')
                            
                            pred = all_samples[-1].reshape(all_samples[-1].shape[0], self.config.model.ImgChnNum*self.config.model.clip_pred,
                                                        self.config.data.image_size, self.config.data.image_size).to(self.config.device)

                            save_image(test_X[0], os.path.join(self.args.log_sample_path, 'X0_{}.png'.format(step)))
                            save_image(pred[0], os.path.join(self.args.log_sample_path, 'pred0_{}.png'.format(step)))
                            del all_samples
                        
                        upsample=nn.Upsample(scale_factor=4, mode='nearest')
                        loss_mv_val = score_func(upsample(mv_recon_t), upsample(sample_mvs_t)).cpu().data.numpy()# 128,24,256,256
                        loss_frame_val = score_func(pred,test_X).cpu().data.numpy()# 128,1,256,256
                        loss_frame_val = np.mean(loss_frame_val, axis=1) # 128,256,256
                        loss_frame_val = np.expand_dims(loss_frame_val,axis=1) # 128,1,256,256

                        loss_mv_val = np.sum(np.sum(np.sum(loss_mv_val, axis=3), axis=2), axis=1)
                        loss_mv_val = (loss_mv_val-np.mean(loss_mv_val))/np.std(loss_mv_val)
                        loss_frame_val = np.sum(np.sum(np.sum(loss_frame_val, axis=3), axis=2), axis=1)
                        loss_frame_val = (loss_frame_val-np.mean(loss_frame_val))/np.std(loss_frame_val)

                        print("mv_socre:",loss_mv_val)
                        print("frame_score:",loss_frame_val)

                        score_final1 = self.config.eval.wr*loss_mv_val + self.config.eval.wp*loss_frame_val  # 算这个的mean和std
                         # input_all_score=[]
                        # h = int((score_final1.shape)[-2]/self.step_size)
                        # w = int((score_final1.shape)[-1]/self.step_size)
                        # for i in range(h):
                        #     for j in range(w):
                        #         patch=score_final1[:,:,i*self.step_size:(i+1)*self.step_size,j*self.step_size:(j+1)*self.step_size]
                        #         patch_score = np.sum(np.sum(np.sum(patch, axis=3), axis=2), axis=1)
                        #         input_all_score.append(patch_score)
                        # input_all_score_array=np.array(input_all_score)
                        # input_scores=np.transpose(input_all_score_array,[1,0])
                        # score_final1 = input_scores.max(1)

                        # mean, std = np.mean(score_final1), np.std(score_final1)
                        # score_final1 = (score_final1 - mean)/std

                        # score_final1 = self.config.eval.wr*loss_mv_val + self.config.eval.wp*loss_frame_val
                        # score_final1 = np.sum(np.sum(np.sum(score_final1, axis=3), axis=2), axis=1)

                        # anomaly scores for each sample
                        for i in range(len(score_final1)):
                            video_index=video_list.index(v_name[i])
                            frame_scores[video_index][pred_frame_test[i]] = score_final1[i] ##the score of corresponding frame

                    # print('framsc1shape:',frame_scores)
                    frame_scores2=[]
                    for k in range(len(video_list)):
                        index=np.flatnonzero(frame_scores[k])##the index of no-zero
                        # print(k)
                        # print('index:',index)
                        score_list=[frame_scores[k][j] for j in index]
                        score_list_final=self.frame_level_result(score_list)##The score of unsampled frames is the average of two adjacent sampled frames
                        frame_scores[k][index[0]:index[-1]+1]=score_list_final
                        # print(frame_scores)
                        frame_scores2.append([score_list_final,index[0],index[-1]])
                    # print('framescore2shape:',frame_scores2)
                    
                    original_frame_scores=frame_scores
                    # ================== Calculate AUC ==============================
                    # load gt labels
                    gt = pickle.load(
                        open(os.path.join(self.config.eval.gt_dir, "%s/ground_truth_demo/gt_label.json" % dataset_name), "rb"))
                    gt_concat = np.concatenate(list(gt.values()), axis=0)

                    new_gt = np.array([])
                    new_frame_scores = np.array([])

                    video_label_num=[]##the number of labels every video
                    frames_idx = 0
                    for cur_video_id in range(METADATA[dataset_name]["testing_video_num"]):
                        cur_video_len = METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]

                        start_idx=frame_scores2[cur_video_id][1]
                        end_idx=frame_scores2[cur_video_id][2]+1
                        gt_video=gt_concat[frames_idx:frames_idx + cur_video_len] #The gt corresponding to each video
                        gt_each_video = gt_video[start_idx:end_idx]
                        scores_each_video = frame_scores2[cur_video_id][0]
                        
                        video_label_num.append(len(scores_each_video))

                        frames_idx += cur_video_len

                        new_gt = np.concatenate((new_gt, gt_each_video), axis=0)
                        new_frame_scores = np.concatenate((new_frame_scores, scores_each_video), axis=0)

                    gt_concat = new_gt
                    frame_scores = new_frame_scores
                    # print(frame_scores)
                    ##only save the best result
                    curves_save_path = os.path.join(self.args.log_path, 'anomaly_curves')
                    auc = save_evaluation_curves_pred(original_frame_scores,frame_scores, gt_concat, curves_save_path,np.array(video_label_num),best_auc)

                    print("auc in step{} is {}".format(step,auc))
                    ##save auc every validation step
                    auc_file = open(txtpath, 'a')
                    auc_file.write( 'checkpoint_{}.pth\tAUC: {}\n'.format(step, auc ))
                    auc_file.close()
                    
                    if auc >= best_auc:
                        states = [
                                scorenet.state_dict(),
                                optimizer.state_dict(),
                                epoch,
                                step,
                                ]
                        if self.config.model.ema:
                            states.append(ema_helper.state_dict())
                        best_auc = auc
                        torch.save(states, os.path.join(self.args.log_path, "best.pth"))
                        logging.info(f"Saving best checkpoint.pth in {self.args.log_path}")
                        val_times = 0
                    else:
                        val_times += 1
                        
                    # Plot graphs
                    try:
                        plot_graphs_process.join()
                    except:
                        pass
                    plot_graphs_process = Process(target=self.plot_graphs)
                    plot_graphs_process.start()

                ###################################
                del test_scorenet

                self.time_elapsed.update(self.convert_time_stamp_to_hrs(str(datetime.timedelta(seconds=(time.time() - self.start_time)))) + self.time_elapsed_prev)

                # Save meters
                if step == 1 or step % self.config.training.val_freq == 0 or step % 1000 == 0 :
                    self.save_meters()
                
                if val_times == self.config.eval.early_end:
                    early_end = True
                    break

            # scheduler.step()
            if early_end:
                break

        # Save model at the very end
        states = [
            scorenet.state_dict(),
            optimizer.state_dict(),
            epoch,
            step,
        ]
        if self.config.model.ema:
            states.append(ema_helper.state_dict())

        logging.info(f"Saving checkpoints in {self.args.log_path}")
        torch.save(states, os.path.join(self.args.log_path, 'checkpoint_{}.pt'.format(step)))
        torch.save(states, os.path.join(self.args.log_path, 'checkpointlast.pt'))
        print("================ Best AUC %.5f ================" % best_auc)
    
    def test(self):
        scorenet = get_model(self.config)
        scorenet = torch.nn.DataParallel(scorenet)

        test_dataset = VideoDataset(self.config.model.ImgChnNum, self.config.model.sampled_mv_num,
                                     self.testset_yuvroot, self.testset_mvroot)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.config.test.batch_size, 
                                 num_workers=self.config.data.num_workers, shuffle=False)
        
        conditional = self.config.data.num_frames_cond > 0
        test_cond = None
        verbose = False
        # Sampler
        sampler = self.get_sampler()
        # MV recon net
        recon_MV_dict = torch.load(self.config.model.recon_MV_pretrained)["model_state_dict"]
        recon_AE = reconAE(num_in_ch=self.config.model.motion_channels*self.config.model.sampled_mv_num, 
                            seq_len=1, features_root=self.config.model.feature_root,
                            skip_ops=self.config.model.skip_ops).to(self.config.device)
        recon_AE = torch.nn.DataParallel(recon_AE)
        recon_AE.load_state_dict(recon_MV_dict, False)
        for param in recon_AE.parameters():
            param.requires_grad = False
        recon_AE.eval()

        states = torch.load(self.config.test.ckpt)
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(scorenet)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(scorenet)
        else:
            scorenet.load_state_dict(states[0])

        scorenet.eval()

        dataset_name = self.config.data.dataset
        score_func = nn.MSELoss(reduction="none")
        # bbox anomaly scores for each frame
        video_list = [name for name in os.listdir(self.testset_yuvroot)]
        video_list.sort()
        #print(video_list)
        frame_scores=[]
        for k in range(len(video_list)):
            m=[0 for i in range((METADATA[dataset_name]["testing_frames_cnt"])[k])]
            frame_scores.append(m)

        best_auc = -1
        for test_data in tqdm(test_loader, desc="Eval:", total=len(test_loader)):
            test_X, sample_mvs_t, pred_frame_test, v_name, mv_last = test_data
            test_X = test_X.to(self.config.device)
            test_X, test_cond, test_cond_mask = conditioning_fn(self.config, test_X, num_frames_pred=self.config.model.clip_pred,
                                                                prob_mask_cond=getattr(self.config.data, 'prob_mask_cond', 0.0),
                                                                prob_mask_future=getattr(self.config.data, 'prob_mask_future', 0.0),
                                                                conditional=conditional)
            
            # MV recon
            mv_recon_t = torch.zeros_like(sample_mvs_t)
            y_ch = self.config.model.motion_channels*self.config.model.sampled_mv_num
            
            for j in range(self.config.model.clip_hist):
                _, reconAE_out = recon_AE(sample_mvs_t[:, y_ch * j:y_ch * (j + 1), :, :])
                mv_recon_t[:, y_ch * j:y_ch * (j + 1), :, :] = reconAE_out
            upsample=nn.Upsample(scale_factor=4, mode='nearest')
            sample_mvs_t1 = upsample(mv_recon_t)
            # sample_mvs_t1 = F.interpolate(mv_recon_t, size=(256, 256), mode='bilinear', align_corners=False)
            # print("samplemv's shape: ",sample_mvs_t.shape)
            
            sample_mvs_t_split = sample_mvs_t1.view(sample_mvs_t1.shape[0],4,6,256,256)
            cond_ip_t = np.zeros((sample_mvs_t1.shape[0], 28,256,256))

            # Initial samples
            n_init_samples = sample_mvs_t1.shape[0]
            channel_num = self.config.model.ImgChnNum*self.config.model.clip_pred
            # channel_num = self.config.model.ImgChnNum + self.config.model.clip_hist*(self.config.model.sampled_mv_num*self.config.model.motion_channels+ self.config.model.ImgChnNum)
            # init_samples_shape = (n_init_samples, self.config.data.channels*self.config.data.num_frames, self.config.data.image_size, self.config.data.image_size)
            init_samples_shape = (n_init_samples, channel_num, self.config.data.image_size, self.config.data.image_size)
            # print("init_sample_shape:", init_samples_shape)

            if self.version == "DDPM" or self.version == "DDIM" :
                if getattr(self.config.model, 'gamma', False):    # 使用gamma分布
                    used_k, used_theta = scorenet.k_cum[0], scorenet.theta_t[0]
                    z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
                    init_samples = z - used_k*used_theta # we don't scale here
                else:
                    init_samples = torch.randn(init_samples_shape, device=self.config.device)  # 使用高斯分布随机产生
            for i in range(4):
                temp = sample_mvs_t_split[:,i,:,:,:]
                # temp = temp.squeeze(1)
                cond_ip_t[:,7*i:7*(i+1),:,:] = torch.cat((test_cond[:,i,:,:].unsqueeze(1).cpu(), temp.cpu()), dim =1)

            cond_ip_t = torch.from_numpy(cond_ip_t).to(self.config.device)

            with torch.no_grad():
                test_hook = None
                test_dsm_loss = anneal_dsm_score_estimation(scorenet, test_X, labels=None, cond=cond_ip_t, cond_mask=test_cond_mask,
                                                            loss_type=getattr(self.config.training, 'loss_type', 'a'),
                                                            gamma=getattr(self.config.model, 'gamma', False),
                                                            L1=getattr(self.config.training, 'L1', False), hook=test_hook,
                                                            all_frames=getattr(self.config.model, 'output_all_frames', False))

            with torch.no_grad():
                # frame sample for prediction
                all_samples = sampler(init_samples, scorenet, cond=cond_ip_t, cond_mask=test_cond_mask,
                                    n_steps_each=self.config.sampling.n_steps_each,
                                    step_lr=self.config.sampling.step_lr, just_beta=False,
                                    final_only=True, denoise=self.config.sampling.denoise,
                                    subsample_steps=getattr(self.config.sampling, 'subsample', None),
                                    clip_before=getattr(self.config.sampling, 'clip_before', True),
                                    verbose=False, log=False, gamma=getattr(self.config.model, 'gamma', False)).to('cpu')
                
                pred = all_samples[-1].reshape(all_samples[-1].shape[0], self.config.model.ImgChnNum*self.config.model.clip_pred,
                                            self.config.data.image_size, self.config.data.image_size).to(self.config.device)

                save_image(test_X[0], os.path.join(self.args.log_sample_path, 'X0.png'))
                save_image(pred[0], os.path.join(self.args.log_sample_path, 'pred0.png'))
                del all_samples
            
            upsample=nn.Upsample(scale_factor=4, mode='nearest')
            loss_mv_val = score_func(upsample(mv_recon_t), upsample(sample_mvs_t)).cpu().data.numpy()# 128,24,256,256
            loss_frame_val = score_func(pred,test_X).cpu().data.numpy()# 128,1,256,256
            loss_frame_val = np.mean(loss_frame_val, axis=1) # 128,256,256
            loss_frame_val = np.expand_dims(loss_frame_val,axis=1) # 128,1,256,256

            loss_mv_val = np.sum(np.sum(np.sum(loss_mv_val, axis=3), axis=2), axis=1)
            # loss_mv_val = (loss_mv_val-np.mean(loss_mv_val))/np.std(loss_mv_val)
            loss_frame_val = np.sum(np.sum(np.sum(loss_frame_val, axis=3), axis=2), axis=1)
            # loss_frame_val = (loss_frame_val-np.mean(loss_frame_val))/np.std(loss_frame_val)

            print("mv_socre:",loss_mv_val)
            print("frame_score:",loss_frame_val)

            score_final1 = self.config.eval.wr*loss_mv_val + self.config.eval.wp*loss_frame_val  # 算这个的mean和std
            # input_all_score=[]
            # h = int((score_final1.shape)[-2]/self.step_size)
            # w = int((score_final1.shape)[-1]/self.step_size)
            # for i in range(h):
            #     for j in range(w):
            #         patch=score_final1[:,:,i*self.step_size:(i+1)*self.step_size,j*self.step_size:(j+1)*self.step_size]
            #         patch_score = np.sum(np.sum(np.sum(patch, axis=3), axis=2), axis=1)
            #         input_all_score.append(patch_score)
            # input_all_score_array=np.array(input_all_score)
            # input_scores=np.transpose(input_all_score_array,[1,0])
            # score_final1 = input_scores.max(1)

            mean, std = np.mean(score_final1), np.std(score_final1)
            score_final1 = (score_final1 - mean)/std

            # score_final1 = self.config.eval.wr*loss_mv_val + self.config.eval.wp*loss_frame_val
            # score_final1 = np.sum(np.sum(np.sum(score_final1, axis=3), axis=2), axis=1)

            # anomaly scores for each sample
            for i in range(len(score_final1)):
                video_index=video_list.index(v_name[i])
                frame_scores[video_index][pred_frame_test[i]] = score_final1[i] ##the score of corresponding frame

        # print('framsc1shape:',frame_scores)
        frame_scores2=[]
        for k in range(len(video_list)):
            index=np.flatnonzero(frame_scores[k])##the index of no-zero
            # print(k)
            # print('index:',index)
            score_list=[frame_scores[k][j] for j in index]
            score_list_final=self.frame_level_result(score_list)##The score of unsampled frames is the average of two adjacent sampled frames
            frame_scores[k][index[0]:index[-1]+1]=score_list_final
            # print(frame_scores)
            frame_scores2.append([score_list_final,index[0],index[-1]])
        # print('framescore2shape:',frame_scores2)
        
        original_frame_scores=frame_scores
        # ================== Calculate AUC ==============================
        # load gt labels
        gt = pickle.load(
            open(os.path.join(self.config.eval.gt_dir, "%s/ground_truth_demo/gt_label.json" % dataset_name), "rb"))
        gt_concat = np.concatenate(list(gt.values()), axis=0)

        new_gt = np.array([])
        new_frame_scores = np.array([])

        video_label_num=[]##the number of labels every video
        frames_idx = 0
        for cur_video_id in range(METADATA[dataset_name]["testing_video_num"]):
            cur_video_len = METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]

            start_idx=frame_scores2[cur_video_id][1]
            end_idx=frame_scores2[cur_video_id][2]+1
            gt_video=gt_concat[frames_idx:frames_idx + cur_video_len] #The gt corresponding to each video
            gt_each_video = gt_video[start_idx:end_idx]
            scores_each_video = frame_scores2[cur_video_id][0]
            
            video_label_num.append(len(scores_each_video))

            frames_idx += cur_video_len

            new_gt = np.concatenate((new_gt, gt_each_video), axis=0)
            new_frame_scores = np.concatenate((new_frame_scores, scores_each_video), axis=0)

        gt_concat = new_gt
        frame_scores = new_frame_scores
        # print(frame_scores)
        ##only save the best result
        curves_save_path = os.path.join(self.args.log_path, 'anomaly_curves')
        auc = save_evaluation_curves_pred(original_frame_scores,frame_scores, gt_concat, curves_save_path,np.array(video_label_num),best_auc)
        print("auc is {}".format(auc))

    def frame_level_result(self, res_prob_list):

        res_prob_list_final=[]
        for k in range(len(res_prob_list)):
            res_prob_list_final.append(res_prob_list[k])
            if k<len(res_prob_list)-1:
                frame_score=(res_prob_list[k]+res_prob_list[k+1])/2
                for m in range(self.test_interval-1):
                    res_prob_list_final.append(frame_score)
        return res_prob_list_final
    
    def load_meters(self):
        meters_pkl = os.path.join(self.args.log_path,  'meters.pkl')
        if not os.path.exists(meters_pkl):
            print(f"{meters_pkl} does not exist! Returning.")
            return False
        with open(meters_pkl, "rb") as f:
            a = pickle.load(f)
        # Load
        self.epochs = a['epochs']
        self.losses_train = a['losses_train']
        self.losses_test = a['losses_test']
        self.lr_meter = a['lr_meter']
        self.grad_norm = a['grad_norm']
        self.time_train = a['time_train']
        self.time_train_prev = a['time_train'].val or 0
        self.time_elapsed = a['time_elapsed']
        self.time_elapsed_prev = a['time_elapsed'].val or 0

    def init_meters(self):
        success = self.load_meters()    
        if not success:
            self.epochs = RunningAverageMeter()
            self.losses_train, self.losses_test = RunningAverageMeter(), RunningAverageMeter()
            self.lr_meter, self.grad_norm = RunningAverageMeter(), RunningAverageMeter()
            self.time_train, self.time_elapsed = RunningAverageMeter(), RunningAverageMeter()
            self.time_train_prev = self.time_elapsed_prev = 0
        
    def only_model_saver(self, model_state_dict, model_path):
        state_dict = {}
        state_dict["model_state_dict"] = model_state_dict

        torch.save(state_dict, model_path)
        print('models {} save successfully!'.format(model_path))

    def save_meters(self):
        meters_pkl = os.path.join(self.args.log_path, 'meters.pkl')
        with open(meters_pkl, "wb") as f:
            pickle.dump({
                'epochs': self.epochs,
                'losses_train': self.losses_train,
                'losses_test': self.losses_test,
                'lr_meter' : self.lr_meter,
                'grad_norm' : self.grad_norm,
                'time_train': self.time_train,
                'time_elapsed': self.time_elapsed,
                },
                f, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_graphs(self):
        # Losses
        plt.plot(self.losses_train.steps, self.losses_train.vals, label='Train')
        plt.plot(self.losses_test.steps, self.losses_test.vals, label='Test')
        plt.xlabel("Steps")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        plt.legend(loc='upper right')
        self.savefig(os.path.join(self.args.log_path, 'loss.png'))
        plt.yscale("log")
        self.savefig(os.path.join(self.args.log_path, 'loss_log.png'))
        plt.clf()
        plt.close()
        # Epochs
        plt.plot(self.losses_train.steps, self.epochs.vals)
        plt.xlabel("Steps")
        plt.ylabel("Epochs")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        self.savefig(os.path.join(self.args.log_path, 'epochs.png'))
        plt.clf()
        plt.close()
        # LR
        plt.plot(self.losses_train.steps, self.lr_meter.vals)
        plt.xlabel("Steps")
        plt.ylabel("LR")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        self.savefig(os.path.join(self.args.log_path, 'lr.png'))
        plt.clf()
        plt.close()
        # Grad Norm
        plt.plot(self.losses_train.steps, self.grad_norm.vals)
        plt.xlabel("Steps")
        plt.ylabel("Grad Norm")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        self.savefig(os.path.join(self.args.log_path, 'grad.png'))
        plt.yscale("log")
        self.savefig(os.path.join(self.args.log_path, 'grad_log.png'))
        plt.clf()
        plt.close()
        # Time train
        plt.plot(self.losses_train.steps, self.time_train.vals)
        plt.xlabel("Steps")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        self.savefig(os.path.join(self.args.log_path, 'time_train.png'))
        plt.clf()
        plt.close()
        # Time elapsed
        plt.plot(self.losses_train.steps[:len(self.time_elapsed.vals)], self.time_elapsed.vals)
        plt.xlabel("Steps")
        plt.grid(True)
        plt.grid(visible=True, which='minor', axis='y', linestyle='--')
        self.savefig(os.path.join(self.args.log_path, 'time_elapsed.png'))
        plt.clf()
        plt.close()
    
    def savefig(self, path, bbox_inches='tight', pad_inches=0.1):
        try:
            plt.savefig(path, bbox_inches=bbox_inches, pad_inches=pad_inches)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            print(sys.exc_info()[0])