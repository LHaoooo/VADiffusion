import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import pickle
import numpy as np
from tqdm import tqdm
from utils.train_util import (
    model_defaults,
    args_to_dict,
    add_dict_to_argparser,
    )
from dataset.YUVvideo_dataset import VideoDataset
from models.MV_reconAE import reconAE
from utils.eval_util import save_evaluation_curves

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
def frame_level_result(res_prob_list):
    res_prob_list_final=[]
    for k in range(len(res_prob_list)):
        res_prob_list_final.append(res_prob_list[k])
        if k<len(res_prob_list)-1:
            frame_score=(res_prob_list[k]+res_prob_list[k+1])/2
            for m in range(test_interval-1):
                res_prob_list_final.append(frame_score)
    return res_prob_list_final

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

def evaluate(args, ckpt_path, testset_yuvroot,testset_mvroot,dataloader_test,best_auc,suffix):
    dataset_name = args.dataset_name
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
    
    device = torch.device("cuda")
    # num_workers = args.num_workers
    eval_dir = os.path.join(args.ckpt_root, args.exp_name, args.eval_root)

    os.makedirs(eval_dir, exist_ok=True)

    model = create_model(**args_to_dict(args, model_defaults().keys())).to(device).eval()
    model = nn.DataParallel(model)

    # load weights
    model_weights = torch.load(ckpt_path)["model_state_dict"]
    model.load_state_dict(model_weights)
    print("load pre-trained success!")

    score_func = nn.MSELoss(reduction="none")#no reduction will be applied.
    #anomaly scores for each frame
    video_list = [name for name in os.listdir(testset_yuvroot)]
    video_list.sort()
    #print(video_list)
    frame_scores=[]
    for k in range(len(video_list)):
        m=[0 for i in range((METADATA[dataset_name]["testing_frames_cnt"])[k])] 
        frame_scores.append(m)  
    
    for ii, test_data in tqdm(enumerate(dataloader_test), desc="Eval: ", total=len(dataloader_test)):
        _, sample_mvs_test,pred_frame_test,v_name,_= test_data
        sample_mvs_test = sample_mvs_test.cuda()

        mv_target_test,out_test = model(sample_mvs_test)
        loss_of_test = score_func(out_test, mv_target_test).cpu().data.numpy()
        scores = np.sum(np.sum(np.sum(loss_of_test, axis=3), axis=2), axis=1) 
        # print(scores)
        # anomaly scores for each sample
        for i in range(len(scores)):
            video_index=video_list.index(v_name[i])
            frame_scores[video_index][pred_frame_test[i]] = scores[i] ##the score of corresponding frame
    # print(frame_scores)
    frame_scores2=[]
    for k in range(len(video_list)):
        index=np.flatnonzero(frame_scores[k])##the index of no-zero
        score_list=[frame_scores[k][j] for j in index]
        score_list_final=frame_level_result(score_list)##The score of unsampled frames is the average of two adjacent sampled frames
        frame_scores[k][index[0]:index[-1]+1]=score_list_final
        frame_scores2.append([score_list_final,index[0],index[-1]])

    # joblib.dump(frame_scores, os.path.join(args.ckpt_root, args.exp_name,args.eval_root,
    #                                             "frame_scores_%s.json" % suffix))
    original_frame_scores=frame_scores
    # ================== Calculate AUC ==============================
    # load gt labels
    gt = pickle.load(
        open(os.path.join(args.gt_dir, "%s/ground_truth_demo/gt_label.json" % dataset_name), "rb"))
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
    #curves_save_path = os.path.join(args.ckpt_root, args.exp_name,args.eval_root, 'anomaly_curves_%s' % suffix)
    ##only save the best result
    curves_save_path = os.path.join(args.ckpt_root, args.exp_name,args.eval_root, 'anomaly_curves')
    auc = save_evaluation_curves(args,original_frame_scores,frame_scores, gt_concat, curves_save_path,np.array(video_label_num),best_auc)

    return auc


if __name__ == '__main__':
    def create_argparser():
        defaults = dict(
            motion_channels = 2,  # MV data channel
            sampled_mv_num = 3,  # the num of sampled mv in one GOP
            ImgChnNum =  1, # channel of I frame UCSD
            # ImgChnNum =  3, # channel of I frame AVe
            num_mvs = 1,
            feature_root = 16,
            skip_conn = True,
            skip_ops = ["none", "none", "none","none"],
            #skip_ops = [ "none", "concat", "concat","concat"],

            # exp settings
            dataset_base_dir =  "/home/Dataset/UCSD_ped/UCSD_ped2",  # UCSD_ped2
            gt_dir = "data",

            ckpt_root = "mv_ckpt",
            log_root = "log",

            dataset_name = "UCSD_ped2",
            exp_name =  "UCSD_ped2_eval",  # MV stack

            eval_root = "eval",
            device_ids = "1,2,3,4",
            seed = 123456,

            pretrained =  False,
            model_savename = "mv_model",

            logevery = 100 , # num of iterations to log
            saveevery = 1 , # num of epoch to save models

            # training setting
            num_epochs = 100,
            batch_size = 32,
            lr = 0.005,
            num_workers = 0,
        )
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        return parser

    args = create_argparser().parse_args()

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)  
    torch.cuda.manual_seed_all(args.seed) 
    # np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    best_auc = -1
    model_save_path = "/home/VADiffusion/mv_ckpt/UCSD_ped2_mv_recon_stack_mv_114514/stackbest.pth"
    testset_yuvroot=os.path.join(args.dataset_base_dir,  "test_recyuv400/")
    testset_mvroot=os.path.join(args.dataset_base_dir,  "testmv_txt/")

    dataset_test = VideoDataset(args.ImgChnNum, args.sampled_mv_num, testset_yuvroot, testset_mvroot, last_mv=True)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=128, num_workers=args.num_workers, shuffle=False)
    with torch.no_grad():
        auc = evaluate(args, model_save_path,testset_yuvroot,testset_mvroot,dataloader_test,best_auc,suffix="best") 
        print(auc)