import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scipy.signal as signal
import joblib
import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel._functions import Scatter

def draw_roc_curve(fpr, tpr, auc, psnr_dir):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.5f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC CURVE')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(psnr_dir, "auc.png"))
    plt.close()


def nonzero_intervals(vec):
    '''
    Find islands of non-zeros in the vector vec
    '''
    if len(vec) == 0:
        return []
    elif not isinstance(vec, np.ndarray):
        vec = np.array(vec)

    tmp1 = (vec == 0) * 1
    tmp = np.diff(tmp1)
    edges, = np.nonzero(tmp)
    edge_vec = [edges + 1]

    if vec[0] != 0:
        edge_vec.insert(0, [0])
    if vec[-1] != 0:
        edge_vec.append([len(vec)])
    edges = np.concatenate(edge_vec)
    return zip(edges[::2], edges[1::2])


def save_evaluation_curves(args,original_frame_scores,scores, labels, curves_save_path, video_frame_nums,best_auc):
    """
    Draw anomaly score curves for each video and the overall ROC figure.
    """
    

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
        #中值滤波
        scores_each_video[video_id] = signal.medfilt(scores_each_video[video_id], kernel_size=17)
        labels_each_video[video_id] = labels[start_idx:start_idx + video_frame_nums[video_id]]

        start_idx += video_frame_nums[video_id]

    truth = []
    preds = []
    for i in range(len(scores_each_video)):
        truth.append(labels_each_video[i])
        preds.append(scores_each_video[i])

    truth = np.concatenate(truth, axis=0)
    preds = np.concatenate(preds, axis=0)
    
    # print(truth)
    # print(preds)
    # preds=np.nan_to_num(preds.astype(np.float64))
    fpr, tpr, roc_thresholds = roc_curve(truth, preds, pos_label=0)
    auroc = auc(fpr, tpr)
    
    if auroc>=best_auc:##save the best result
        joblib.dump(original_frame_scores, os.path.join(args.ckpt_root, args.exp_name,args.eval_root,
                                                "frame_scores_best.json" ))
        if not os.path.exists(curves_save_path):
            os.mkdir(curves_save_path)
        # draw ROC figure
        draw_roc_curve(fpr, tpr, auroc, curves_save_path)
        for i in sorted(scores_each_video.keys()):
            plt.figure()

            x = range(0, len(scores_each_video[i]))
            plt.xlim([x[0], x[-1] + 5])

            # anomaly scores
            plt.plot(x, 1-scores_each_video[i], color="blue", lw=2, label="Anomaly Score")

            # abnormal sections
            lb_one_intervals = nonzero_intervals(labels_each_video[i])
            for idx, (start, end) in enumerate(lb_one_intervals):
                plt.axvspan(start, end, alpha=0.5, color='red',
                            label="_" * idx + "Anomaly Intervals")

            plt.xlabel('Frames Sequence')
            plt.title('Test video #%d' % (i + 1))
            plt.legend(loc="upper left")
            plt.savefig(os.path.join(curves_save_path, "anomaly_curve_%d.png" % (i + 1)))
            plt.close()

    return auroc

def save_evaluation_curves_pred(original_frame_scores,scores, labels, curves_save_path, video_frame_nums,best_auc):
    """
    Draw anomaly score curves for each video and the overall ROC figure.
    """

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
        #中值滤波
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
    
    # print(truth)
    # print(preds)
    # preds=np.nan_to_num(preds.astype(np.float64))
    fpr, tpr, roc_thresholds = roc_curve(truth, preds, pos_label=0)
    auroc = auc(fpr, tpr)
    
    # calculate EER
    # fnr = 1 - tpr
    # eer1 = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    # eer2 = fnr[np.nanargmin(np.absolute(fnr - fpr))]

    if auroc>=best_auc:##save the best result
        if not os.path.exists(curves_save_path):
            os.mkdir(curves_save_path)
        joblib.dump(original_frame_scores, os.path.join(curves_save_path,
                                                "frame_scores_best.json" ))
        # draw ROC figure
        draw_roc_curve(fpr, tpr, auroc, curves_save_path)
        for i in sorted(scores_each_video.keys()):
            plt.figure(figsize=(10,2))

            x = range(0, len(scores_each_video[i]))
            plt.xlim([x[0], x[-1] + 5])

            # auc for each video
            truth1 = []
            preds1 = []
            truth1.append(labels_each_video[i])
            preds1.append(scores_each_video[i])
            truth1 = np.concatenate(truth1, axis=0)
            preds1 = np.concatenate(preds1, axis=0)
            fpr1, tpr1, roc_thresholds = roc_curve(truth1, preds1, pos_label=0)
            auroc1 = auc(fpr1, tpr1)

            # anomaly scores
            plt.plot(x, 1-scores_each_video[i], color="darkviolet", lw=2, label=['AUROC=%0.2f%%'% (100*auroc1),'Anomaly Score'] )

            # abnormal sections
            lb_one_intervals = nonzero_intervals(labels_each_video[i])
            for idx, (start, end) in enumerate(lb_one_intervals):
                plt.axvspan(start, end, alpha=0.5, color='pink',
                            label="_" * idx + "Anomaly Intervals")

            plt.xlabel('Frames Sequence',fontsize=12)
            # plt.title('Test video #%d' % (i + 1))
            # plt.legend(loc="upper left")
            plt.legend(loc="best")
            plt.savefig(os.path.join(curves_save_path, "anomaly_curve_%d.png" % (i + 1)))
            plt.close()

    return auroc, delta_s

def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None

def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

class BalancedDataParallel(DataParallel):
    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids)
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]
        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)

        # print('bsz: ', bsz)
        # print('num_dev: ', num_dev)
        # print('gpu0_bsz: ', gpu0_bsz)
        # print('bsz_unit: ', bsz_unit)
        # print('chunk_sizes: ', chunk_sizes)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)