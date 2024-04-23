import pickle
import scipy.signal as signal
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
from tqdm import tqdm
from collections import OrderedDict
# from model.utils import DataLoader
# from model.final_future_prediction_shanghai import *
from model.final_future_prediction_avenue import *
from utils import *
import glob
import argparse
from YUVdataset import VideoDataset
import joblib
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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

def MNADEval(model=None):
    parser = argparse.ArgumentParser(description="DMAD")
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
    parser.add_argument('--dim', type=int, default=512, help='channel dimension of the memory items')
    parser.add_argument('--msize', type=int, default=1000, help='number of the memory items')
    parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
    parser.add_argument('--dataset_type', type=str, default='shanghai', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--dataset_path', type=str, default='/home/shanghaitech', help='directory of data')
    parser.add_argument('--model_dir', type=str,  default='./modelzoo/shanghai.pth', help='directory of model')

    args = parser.parse_args()

    torch.manual_seed(2020)
    os.environ["CUDA_VISIBLE_DEVICES"]= "1"
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
    
    testset_yuvroot=os.path.join(args.dataset_path, "test_recyuv/")
    test_dataset = VideoDataset(args.c, testset_yuvroot)  
    test_batch = data.DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, 
                                num_workers=args.num_workers_test, shuffle=False, drop_last=False)
    
    # Loading dataset
    # test_dataset = DataLoader(test_folder, transforms.Compose([
    #             transforms.ToTensor(),
    #             ]), resize_height=args.h, resize_width=args.w, time_step=4, c=args.c)

    # test_batch = data.DataLoader(test_dataset, batch_size = args.test_batch_size,
    #                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    # Loading the trained model
    if model is None: # if not training, we give a exist model and params path
        model = convAE(args.c, 5, args.msize, args.dim)
        try:
            model.load_state_dict(torch.load(args.model_dir).state_dict(),strict=False)
        except:
            model.load_state_dict(torch.load(args.model_dir),strict=False)
        model.load_bkg(get_bkg(args.w))
        model.cuda()
    else:
        model = convAE(args.c, 5, args.msize, args.dim)
        model_dir = './exp/shanghai/log/temp.pth'
        try:
            model.load_state_dict(torch.load(model_dir).state_dict())
        except:
            model.load_state_dict(torch.load(model_dir))
        # model.load_bkg(get_bkg(args.w))
        model.cuda()

    model.eval()
    video_list = [name for name in os.listdir(testset_yuvroot)]
    video_list.sort()
    frame_scores=[]
    for k in range(len(video_list)):
        m=[0 for i in range((METADATA['shanghaitech']["testing_frames_cnt"])[k])]
        frame_scores.append(m)

    list1 = {}
    list2 = {}
    list3 = {}

    print('Evaluation of', args.dataset_type)

    # Setting for video anomaly detection
    for video in sorted(video_list):
        video_name = video.split('t')[-1]
        list1[video_name] = []
        list2[video_name] = []
        list3[video_name] = []

    with torch.no_grad():
        for k,(imgs, pred_frame_test, v_name) in enumerate(tqdm(test_batch)):
            imgs = Variable(imgs).cuda()
            outputs = model.forward(imgs[:, 0:12, :, :], True)
            err_map = model.loss_function(imgs[:, -3:, :, :], *outputs, True)
            latest_losses = model.latest_losses()

            loss_frame_val1 = latest_losses['err1']
            loss_frame_val2 = latest_losses['mse2']
            loss_frame_val3 = latest_losses['grad']
            loss_frame_val = 0.2*loss_frame_val1+0.4*loss_frame_val2+0.6*loss_frame_val3
            loss_frame_val = loss_frame_val.item()

            for i in range(1):
                video_index=video_list.index(v_name[i])
                frame_scores[video_index][pred_frame_test[i]] = loss_frame_val

    frame_scores2=[]
    for k in range(len(video_list)):
        index=np.flatnonzero(frame_scores[k])##the index of no-zero
        score_list=[frame_scores[k][j] for j in index]
        score_list_final=frame_level_result(score_list)##The score of unsampled frames is the average of two adjacent sampled frames
        frame_scores[k][index[0]:index[-1]+1]=score_list_final
        frame_scores2.append([score_list_final,index[0],index[-1]])
        
    original_frame_scores=frame_scores
    # ================== Calculate AUC ==============================
    # load gt labels
    gt = pickle.load(
        open('/home/VADiffusion/data/shanghaitech/ground_truth_demo/gt_label.json', "rb"))
    gt_concat = np.concatenate(list(gt.values()), axis=0)

    new_gt = np.array([])
    new_frame_scores = np.array([])

    video_label_num=[]##the number of labels every video
    frames_idx = 0
    for cur_video_id in range(METADATA['shanghaitech']["testing_video_num"]):
        cur_video_len = METADATA['shanghaitech']["testing_frames_cnt"][cur_video_id]

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
    auc,delta_s = save_evaluation_curves_pred(original_frame_scores,frame_scores, gt_concat,np.array(video_label_num))

    print('The result of ', args.dataset_type)
    print('AUC: ', auc*100, '%')

    return auc*100

def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    max_ele = np.max(psnr_list)
    min_ele = np.min(psnr_list)
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], max_ele, min_ele))
        
    return anomaly_score_list

def frame_level_result(res_prob_list):

    res_prob_list_final=[]
    for k in range(len(res_prob_list)):
        res_prob_list_final.append(res_prob_list[k])
        if k<len(res_prob_list)-1:
            frame_score=(res_prob_list[k]+res_prob_list[k+1])/2
            for m in range(4-1):
                res_prob_list_final.append(frame_score)
    return res_prob_list_final

def save_evaluation_curves_pred(original_frame_scores,scores, labels, video_frame_nums):
    scores = scores.flatten()
    labels = labels.flatten()

    scores_each_video = {}
    labels_each_video = {}

    start_idx = 0
    for video_id in range(len(video_frame_nums)):
        scores_each_video[video_id] = scores[start_idx:start_idx + video_frame_nums[video_id]]
        res_prob=np.array(scores_each_video[video_id])
        res_prob_norm = res_prob - res_prob.min()
        res_prob_norm = 1-(res_prob_norm / res_prob_norm.max())
        scores_each_video[video_id]=res_prob_norm.tolist()
        scores_each_video[video_id] = signal.medfilt(scores_each_video[video_id], kernel_size=17)
        labels_each_video[video_id] = labels[start_idx:start_idx + video_frame_nums[video_id]]

        start_idx += video_frame_nums[video_id]

    truth = []
    preds = []
    anomlys = []
    normals = []
    for i in range(len(scores_each_video)):
        truth.append(labels_each_video[i])
        preds.append(scores_each_video[i])
        anomly_idx = [i for i, x in enumerate(labels_each_video[i]) if x != 0]
        normal_idx = [i for i, x in enumerate(labels_each_video[i]) if x == 0]
        for j in anomly_idx:
            anomlys.append(scores_each_video[i][j])
        for k in normal_idx:
            normals.append(scores_each_video[i][k])

    truth = np.concatenate(truth, axis=0)
    preds = np.concatenate(preds, axis=0)
    delta_s = np.mean(anomlys) - np.mean(normals)
    fpr, tpr, roc_thresholds = roc_curve(truth, preds, pos_label=0)
    auroc = auc(fpr, tpr)
    
    # calculate EER
    # fnr = 1 - tpr
    # eer1 = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    # eer2 = fnr[np.nanargmin(np.absolute(fnr - fpr))]

    return auroc, delta_s

if __name__=='__main__':
    MNADEval()

#     data = np.load("./exp.bak/res_list_shanghai.npz")

#     anomaly_list1 = conf_avg(np.array(data['arr_0']), 55, "average")
#     anomaly_list2 = conf_avg(np.array(data['arr_2']), 55, "average")
#     anomaly_list3 = conf_avg(np.array(data['arr_3']), 55, "average")

#     hyp_alpha = [0.2, 0.6, 0.4]
    
#     comb = np.array(anomaly_list1) * hyp_alpha[0] + np.array(anomaly_list2) * hyp_alpha[1] + np.array(anomaly_list3) * hyp_alpha[2]

#     accuracy = filter(np.array(conf_avg(comb, 55)), data['arr_5'][0], "..\\..\\shanghai\\testing\\frames")
#     print('AUC: ', accuracy * 100, '%; (alpha = ', hyp_alpha, ')')
