# writer ： Liuhao
# create_time ： 2023/5/21 21:00
# file_name：frame_pred_eval.py

import argparse
import os
import torch
import cv2
import joblib
import pickle
import pdb
import numpy as np
from models.ema import EMAHelper
from torch.distributions.gamma import Gamma
from utils.ncsn_runner import get_model
np.seterr(divide='ignore',invalid='ignore')
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml
from dataset.YUVvideo_dataset import VideoDataset
from utils.eval_util import save_evaluation_curves
from PIL import Image,ImageDraw
import imageio

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

test_interval=4
step_size=16


def frame_level_result(res_prob_list):
    res_prob_list_final=[]
    for k in range(len(res_prob_list)):
        res_prob_list_final.append(res_prob_list[k])
        if k<len(res_prob_list)-1:
            frame_score=(res_prob_list[k]+res_prob_list[k+1])/2
            for m in range(test_interval-1):
                res_prob_list_final.append(frame_score)
    return res_prob_list_final

def evaluate(config, ckpt_path, testset_yuvroot,testset_mvroot,dataloader_test,training_stats_path,best_auc,suffix):
    dataset_name = config.data.dataset
    os.environ['CUDA_VISIBLE_DEVICES'] = config.training.device_ids
    device = torch.device("cuda")
    num_workers = config.data.num_workers
    eval_dir = os.path.join(config.training.ckpt_dir, config.eval.exp_name, config.eval.eval_root)
    version = getattr(config.model, 'version', "DDPM")

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    scorenet = get_model(config)
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(scorenet)
    
    if config.model.ema:
        test_scorenet = ema_helper.ema_copy(scorenet)
    else:
        test_scorenet = scorenet
    
    net = scorenet.module if hasattr(scorenet, 'module') else scorenet
    
    # Initial samples
    n_init_samples = max(32, config.training.batch_size)
    # init_samples_shape = (n_init_samples, self.config.data.channels*self.config.data.num_frames, self.config.data.image_size, self.config.data.image_size)
    init_samples_shape = (n_init_samples, config.model.ImgChnNum*config.model.clip_pred, config.data.image_size, config.data.image_size)
    if version == "DDPM" or version == "DDIM" :
        if getattr(config.model, 'gamma', False):    # 使用gamma分布
            used_k, used_theta = net.k_cum[0], net.theta_t[0]
            z = Gamma(torch.full(init_samples_shape, used_k), torch.full(init_samples_shape, 1 / used_theta)).sample().to(self.config.device)
            init_samples = z - used_k*used_theta # we don't scale here
        else:
            init_samples = torch.randn(init_samples_shape, device=config.device)  # 使用高斯分布随机产生

    

    model_weights = torch.load(ckpt_path)["model_state_dict"]
    scorenet.load_state_dict(model_weights)
    # print("load pre-trained success!")

    #  get training stats
    if training_stats_path is not None:
        training_scores_stats = torch.load(training_stats_path)
        
        mean, std = np.mean(training_scores_stats["scores_training_stats"]), \
                                np.std(training_scores_stats["scores_training_stats"])

    score_func = nn.MSELoss(reduction="none")


    # bbox anomaly scores for each frame
    video_list = [name for name in os.listdir(testset_yuvroot)]
    video_list.sort()
    #print(video_list)
    frame_scores=[]
    for k in range(len(video_list)):
        m=[0 for i in range((METADATA[dataset_name]["testing_frames_cnt"])[k])]
        frame_scores.append(m)
    
    for test_data in tqdm(dataloader_test, desc="Eval: ", total=len(dataloader_test)):

        sample_frames_test, sample_mvs_test, pred_frame_test, v_name,mv_last = test_data
        sample_frames_test = sample_frames_test.to(device)
        sample_mvs_test = sample_mvs_test.to(device)
        upsample=nn.Upsample(scale_factor=4, mode='nearest')

        out_test = scorenet(sample_frames_test, sample_mvs_test, mode="test")
        
        loss_mv_test = score_func(upsample(out_test["mv_recon"]), upsample(out_test["mv_target"])).cpu().data.numpy()
        loss_frame_test = score_func(out_test["frame_pred"], out_test["frame_target"]).cpu().data.numpy()
        
        loss_frame_test = np.mean(loss_frame_test, axis=1)
        loss_frame_test=np.expand_dims(loss_frame_test,axis=1)
        
        loss_scores = config["w_r"] * loss_mv_test + config["w_p"] * loss_frame_test
        #scores_area=get_step_score_patch(loss_scores,step_size)
        scores_area=get_step_score_area_weight(loss_scores,step_size,mv_last)
        #scores_area=get_step_score_area_weight_vis(loss_scores,step_size,mv_last,np.array(pred_frame_test), np.array(v_name))
        #scores_area=get_step_score_mv_area_vis(loss_scores,mv_last,np.array(pred_frame_test), np.array(v_name))
        #scores_area=get_step_score_mv_area(loss_scores,mv_last)
        #scores_area=get_step_score_mv_area_centroid(loss_scores,mv_last,step_size)
        #scores_area=get_step_score_mv_area_centroid_vis(loss_scores,mv_last,step_size,np.array(pred_frame_test), np.array(v_name))
        
        if training_stats_path is not None:
            # mean-std normalization
            scores = (scores_area - mean) / std

        # anomaly scores for each sample
        for i in range(len(scores)):
            video_index=video_list.index(v_name[i])
            frame_scores[video_index][pred_frame_test[i]] = scores[i] ##the score of corresponding frame

    frame_scores2=[]
    for k in range(len(video_list)):
        index=np.flatnonzero(frame_scores[k])##the index of no-zero
        score_list=[frame_scores[k][j] for j in index]
        score_list_final=frame_level_result(score_list)##The score of unsampled frames is the average of two adjacent sampled frames
        frame_scores[k][index[0]:index[-1]+1]=score_list_final
        frame_scores2.append([score_list_final,index[0],index[-1]])

    # joblib.dump(frame_scores, os.path.join(config["ckpt_root"], config["exp_name"],config["eval_root"],
    #                                             "frame_scores_%s.json" % suffix))
    original_frame_scores=frame_scores
    # ================== Calculate AUC ==============================
    # load gt labels
    gt = pickle.load(
        open(os.path.join(config["gt_dir"], "%s/ground_truth_demo/gt_label.json" % dataset_name), "rb"))
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
    #pdb.set_trace()
    #curves_save_path = os.path.join(config["ckpt_root"], config["exp_name"],config["eval_root"], 'anomaly_curves_%s' % suffix)
    ##only save the best result
    curves_save_path = os.path.join(config["ckpt_root"], config["exp_name"],config["eval_root"], 'anomaly_curves')
    auc = save_evaluation_curves(config,original_frame_scores,frame_scores, gt_concat, curves_save_path,np.array(video_label_num),best_auc)

    return auc


if __name__ == '__main__':
    from predAE_train_multimv_area import cal_training_stats
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", type=str,
                        default="./pretrained_ckpts/ped2_HF2VAD_99.31.pth",
                        help='path to pretrained weights')
    parser.add_argument("--cfg_file", type=str,
                        default="./pretrained_ckpts/ped2_HF2VAD_99.31_cfg.yaml",
                        help='path to pretrained model configs')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.cfg_file))

    # dataset_name = config["dataset_name"]
    # dataset_base_dir = config["dataset_base_dir"]
    # trainset_yuvroot=os.path.join(dataset_base_dir, "train_recyuv400/")
    # testset_yuvroot=os.path.join(dataset_base_dir,  "test_recyuv400/")
    # trainset_mvroot=os.path.join(dataset_base_dir,  "trainmv_txt/")
    # testset_mvroot=os.path.join(dataset_base_dir,  "testmv_txt/")

    dataset_name = config["dataset_name"]
    dataset_base_dir = config["dataset_base_dir"]
    trainset_yuvroot=os.path.join(dataset_base_dir, "train19_recyuv/")
    testset_yuvroot=os.path.join(dataset_base_dir,  "test19_recyuv/")
    trainset_mvroot=os.path.join(dataset_base_dir,  "train19mv_txt/")
    testset_mvroot=os.path.join(dataset_base_dir,  "test19mv_txt/")
    
    #os.makedirs(os.path.join(config["ckpt_root"], config["exp_name"],config["eval_root"]), exist_ok=True)
    exp_path = os.path.join(config["ckpt_root"], config["exp_name"])
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    training_stat_path = os.path.join(config["ckpt_root"], config["exp_name"], "training_stats-test.npy")
    dataset = VideoDataset(config["model_paras"]["ImgChnNum"],config["model_paras"]["sampled_mv_num"],trainset_yuvroot,trainset_mvroot)
    dataloader = DataLoader(dataset=dataset, batch_size=config["batchsize"], num_workers=config["num_workers"], shuffle=True)
    dataset_test = VideoDataset(config["model_paras"]["ImgChnNum"],config["model_paras"]["sampled_mv_num"],testset_yuvroot,testset_mvroot)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=128, num_workers=0, shuffle=False)
    
    cal_training_stats(config, args.model_save_path, dataloader,training_stat_path) 
    with torch.no_grad():
        auc = evaluate(config, args.model_save_path,
                                testset_yuvroot,testset_mvroot,dataloader_test,
                                training_stat_path,best_auc=0,
                                suffix="best")
        print(auc)
