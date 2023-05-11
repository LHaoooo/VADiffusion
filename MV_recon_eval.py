import os
import torch
from torch.utils.data import DataLoader
import cv2
import torch.nn as nn
import numpy as np
import yaml
import joblib
import pickle
import gc
import numpy as np
import pdb
from tqdm import tqdm
from MV_recon_train import create_model

from dataset.YUVvideo_dataset import VideoDataset
from models.MV_reconAE import reconAE
from utils.eval_util import save_evaluation_curves
from utils.train_util import args_to_dict, model_defaults

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

def evaluate(args, ckpt_path, testset_yuvroot,testset_mvroot,dataloader_test,best_auc,suffix):
    dataset_name = args.dataset_name
    # device_ids = args.device_ids

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids
    # if torch.cuda.is_available() is False:
    #     raise EnvironmentError('not find GPU device for training.')
    
    device = torch.device("cuda")
    num_workers = args.num_workers
    eval_dir = os.path.join(args.ckpt_root, args.exp_name, args.eval_root)

    os.makedirs(eval_dir, exist_ok=True)

    model = create_model(**args_to_dict(args, model_defaults().keys())).to(device).eval()

    # load weights
    model_weights = torch.load(ckpt_path)["model_state_dict"]
    model.load_state_dict(model_weights, False)
    print("load pre-trained success!")

    score_func = nn.MSELoss(reduction="none")#no reduction will be applied.(维度不减少)

    #anomaly scores for each frame
    video_list = [name for name in os.listdir(testset_yuvroot)]
    video_list.sort()
    #print(video_list)
    frame_scores=[]
    for k in range(len(video_list)):
        m=[0 for i in range((METADATA[dataset_name]["testing_frames_cnt"])[k])] # 给测试视频每一帧打分为0
        frame_scores.append(m)
    
    for ii, test_data in tqdm(enumerate(dataloader_test), desc="Eval: ", total=len(dataloader_test)):
        _, sample_mvs_test,pred_frame_test,v_name,_= test_data
        sample_mvs_test = sample_mvs_test.cuda()

        mv_target_test,out_test = model(sample_mvs_test)
        loss_of_test = score_func(out_test, mv_target_test).cpu().data.numpy()
        scores = np.sum(np.sum(np.sum(loss_of_test, axis=3), axis=2), axis=1)  # 这个scores是关于MV重构误差的

        # anomaly scores for each sample
        for i in range(len(scores)):
            video_index=video_list.index(v_name[i])
            frame_scores[video_index][pred_frame_test[i]] = scores[i] ##the score of corresponding frame
    
    frame_scores2=[]
    for k in range(len(video_list)):
        index=np.flatnonzero(frame_scores[k])##the index of no-zero 每个测试视频中MSE不是0的帧
        score_list=[frame_scores[k][j] for j in index]
        score_list_final=frame_level_result(score_list)##The score of unsampled frames is the average of two adjacent sampled frames
        frame_scores[k][index[0]:index[-1]+1]=score_list_final  # 把分数赋给所有帧
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
    model_save_path = "./multimv_ckpt/avenue_recon_multimv=3_train2/best.pth"
    cfg_file = "./multimv_ckpt/avenue_recon_multimv=3_train2/log/recon_cfg.yaml"
    config = yaml.safe_load(open(cfg_file))
    
    
    dataset_name = config["dataset_name"]
    dataset_base_dir = config["dataset_base_dir"]
    num_workers = config["num_workers"]
    best_auc=0
    
    testset_yuvroot=os.path.join(dataset_base_dir,  "test19_recyuv/")
    testset_mvroot=os.path.join(dataset_base_dir,  "test19mv_txt/")
    dataset_test = VideoDataset(config["model_paras"]["ImgChnNum"],config["model_paras"]["sampled_mv_num"],testset_yuvroot,testset_mvroot, last_mv=True)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=128, num_workers=num_workers, shuffle=False)
    with torch.no_grad():
        auc = evaluate(config, model_save_path,testset_yuvroot,testset_mvroot,dataloader_test,best_auc,suffix="best") 
        print(auc)