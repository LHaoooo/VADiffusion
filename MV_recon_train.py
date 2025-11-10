# writer ： Liuhao
# create_time ： 2023/5/10 13:40
# file_name：MV_recon_train.py

'''
Train an AE model to reconstruct MVs
'''

import os
import torch
import argparse
import xlwt
import shutil
from tqdm import tqdm
from torch import nn,optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataset.YUVvideo_dataset import VideoDataset, img_batch_tensor2numpy
from models.MV_reconAE import reconAE
import MV_recon_eval as reconAE_eval
from utils.train_util import (
    model_defaults,
    args_to_dict,
    add_dict_to_argparser,
    only_model_saver,
    saver,
    visualize_sequences,
    weights_init_kaiming,
    )


def main():
    args = create_argparser().parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
    if torch.cuda.is_available() is False:
        raise EnvironmentError('not find GPU device for training.')
    
    world_size = torch.cuda.device_count()
    args.batch_size *= world_size
    args.lr *= world_size
    print("args_agument:",args)

    os.environ['PYTHONHASHSEED'] = str(args.seed)  
    torch.manual_seed(args.seed)    
    torch.cuda.manual_seed_all(args.seed)   
    # np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    paths = dict(
        ckpt_dir = "%s/%s" % (args.ckpt_root, args.exp_name),
        log_dir = "%s/%s/%s" % (args.ckpt_root, args.exp_name, args.log_root)
    )
    if not os.path.exists(paths["ckpt_dir"]):
        os.makedirs(paths["ckpt_dir"])
    if not os.path.exists(paths["log_dir"]):
        os.makedirs(paths["log_dir"])
    txtpath=os.path.join(paths["ckpt_dir"], 'auc.txt')  # the file to save auc
    if (os.path.exists(txtpath)) :
        os.remove(txtpath)

    batch_size = args.batch_size
    epochs = args.num_epochs
    num_workers = args.num_workers
    device = torch.device("cuda")

    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    sheet = book.add_sheet('train_loss',cell_overwrite_ok=True)
    loss_savepath=paths["ckpt_dir"]+"/"+str(args.dataset_name)+"_batchsize="+str(batch_size)+"_learningrate="+str(args.lr)+'_train_loss.xls'

    model = create_model(**args_to_dict(args, model_defaults().keys()))
    model = nn.DataParallel(model.to(device))
    recon_loss = nn.MSELoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-7, weight_decay=0.0)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.5)

    step = 0
    epoch_last = 0

    if not args.pretrained:
        model.apply(weights_init_kaiming)
    else:
        assert (args.pretrained is not None)
        model_state_dict = torch.load(args.pretrained)["model_state_dict"]
        model.load_state_dict(model_state_dict)
        print('pretrained models loaded!', epoch_last)
    
    writer = SummaryWriter(paths["log_dir"])

    # Training
    best_auc = -1

    trainset_yuvroot = os.path.join(args.dataset_base_dir, "train_recyuv400/")
    testset_yuvroot = os.path.join(args.dataset_base_dir, "test_recyuv400/")

    trainset_mvroot = os.path.join(args.dataset_base_dir,  "trainmv_txt/")
    testset_mvroot = os.path.join(args.dataset_base_dir,  "testmv_txt/")

    dataset = VideoDataset(args.ImgChnNum, args.sampled_mv_num, trainset_yuvroot, trainset_mvroot, last_mv = True)  # 第一次用的是TRUE
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    dataset_test = VideoDataset(args.ImgChnNum, args.sampled_mv_num, testset_yuvroot, testset_mvroot, last_mv = True)# 第一次训练用的是TRUE
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=128, num_workers=num_workers, shuffle=False)

    for epoch in range(epoch_last, epochs + epoch_last):  
        total_train_loss = 0
        for _, train_data in tqdm(enumerate(dataloader),
                                  desc="Training Epoch %d" % (epoch + 1),total=len(dataloader)):
            model.train()

            sample_frames, sample_mvs,_,_,_ = train_data
            print(sample_frames.shape)
            print(sample_mvs.shape) # 128*6*64*64
            sample_mvs = sample_mvs.to(device)

            mv_target,out = model(sample_mvs)
            loss_recon = recon_loss(out, mv_target)
            optimizer.zero_grad()
            loss_recon.backward()
            optimizer.step()

            total_train_loss += loss_recon

            if step % args.logevery == args.logevery - 1:
                print("[Step: {}/ Epoch: {}]: Loss: {:.4f} ".format(step + 1, epoch + 1, loss_recon))

                writer.add_scalar('loss_recon/train', loss_recon, global_step=step + 1)

                num_vis = 6
                writer.add_figure("img/train_sample_mvs",
                                    visualize_sequences(
                                        img_batch_tensor2numpy(mv_target.cpu()[:num_vis, :, :, :]),
                                        seq_len=mv_target.size(1) // 2,
                                        return_fig=True),
                                    global_step=step + 1)
                writer.add_figure("img/train_output",
                                    visualize_sequences(img_batch_tensor2numpy(
                                        out.detach().cpu()[:num_vis, :, :, :]),
                                        seq_len=out.size(1) // 2,
                                        return_fig=True),
                                    global_step=step + 1)
                writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step=step + 1)

            step += 1
        print(total_train_loss.item()) 
        sheet.write(epoch,0,total_train_loss.item())
        book.save(loss_savepath)
        scheduler.step()

        if epoch % args.saveevery == args.saveevery - 1:
            model_save_path = os.path.join(paths["ckpt_dir"], args.model_savename)
            saver(model.state_dict(), optimizer.state_dict(), model_save_path, epoch + 1, step, max_to_save=5)

            # evaluation
            with torch.no_grad():
                auc = reconAE_eval.evaluate(args, model_save_path + "-%d" % (epoch + 1)+'.pth',
                                                testset_yuvroot, testset_mvroot, dataloader_test, best_auc,
                                                suffix=str(epoch + 1))
                ##save auc every epoch
                auc_file = open(txtpath, 'a')
                auc_file.write( 'model-{}.pt\tAUC: {}\n'.format(epoch+1, auc ))
                auc_file.close()
                sheet.write(epoch,1,auc)
                book.save(loss_savepath)
                
                if auc >= best_auc:
                    best_auc = auc
                    only_model_saver(model.state_dict(), os.path.join(paths["ckpt_dir"], "stackbest.pth"))

                writer.add_scalar("auc", auc, global_step=epoch + 1)
    writer.close()
    print("================ Best AUC %.5f ================" % best_auc)


def create_argparser():
    defaults = dict(
        # model settings
        motion_channels = 2,  # MV data channel
        # motion_channels = 8,  # 4 MV
        sampled_mv_num = 3,  # the num of sampled mv in one GOP
        # ImgChnNum =  1, # channel of I frame UCSD
        ImgChnNum =  3, # channel of I frame AVe
        num_mvs = 1,
        feature_root = 16,
        skip_conn = True,
        skip_ops = ["none", "none", "none","none"],
        #skip_ops = [ "none", "concat", "concat","concat"],

        # exp settings
        # dataset_base_dir =  "/home/Dataset/UCSD_ped/UCSD_ped2",  # UCSD_ped2
        dataset_base_dir =  "/home/Dataset/Avenue",  # Avenue
        gt_dir = "data",

        ckpt_root = "mv_ckpt",
        log_root = "log",

        # dataset_name = "UCSD_ped2",
        # exp_name =  "UCSD_ped2_mv_recon",  # MV plus
        # exp_name = "UCSD_ped2_mv_recon_stack_mv_123456",  # MV stack
        dataset_name = "avenue",
        exp_name = "avenue_mv_recon_stack_mv_123456_pretrain",  # MV stack

        eval_root = "eval",
        device_ids = "1",
        seed = 123456,

        pretrained =  False,
        # pretrained = "/home/VADiffusion/mv_ckpt/avenue_mv_recon_stack_mv_123456/stackbest.pth",
        model_savename = "mv_model",

        logevery = 100 , # num of iterations to log
        saveevery = 1 , # num of epoch to save models

        # training setting
        num_epochs = 200,
        batch_size = 32,
        lr = 0.005,
        num_workers = 0,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def create_model(
    motion_channels,
    sampled_mv_num,
    num_mvs,
    feature_root,
    skip_ops,
):
    return reconAE(
        num_in_ch = motion_channels*sampled_mv_num,  # stack MV
        # num_in_ch = motion_channels,  # plus MV
        seq_len = num_mvs, 
        features_root = feature_root, 
        skip_ops = skip_ops,
    )

if __name__ == "__main__":
    main()