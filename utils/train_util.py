# writer ： Liuhao
# create_time ： 2023/5/10 14:00
# file_name：train_util.py

import glob
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse

def model_defaults():
    '''
    defaults for MV reconstruction AE training
    '''
    return dict(
        motion_channels = 2,
        sampled_mv_num = 3, #the num of sampled mv in one GOP 
        # ImgChnNum =  1, #channel of I frame
        num_mvs = 1,
        feature_root = 16,
        # skip_conn = True,
        skip_ops = ["none", "none", "none","none"],
        #skip_ops = [ "none", "concat", "concat","concat"],
    )

def saver(model_state_dict, optimizer_state_dict, model_path, epoch, step, max_to_save=8):
    total_models = glob.glob(model_path + '*')
    if len(total_models) >= max_to_save:
        total_models.sort()
        os.remove(total_models[0])

    state_dict = {}
    state_dict["model_state_dict"] = model_state_dict
    state_dict["optimizer_state_dict"] = optimizer_state_dict
    state_dict["step"] = step

    torch.save(state_dict, model_path + '-' + str(epoch)+'.pth')
    print('models {} save successfully!'.format(model_path + '-' + str(epoch)+'.pth'))

def only_model_saver(model_state_dict, model_path):
    state_dict = {}
    state_dict["model_state_dict"] = model_state_dict

    torch.save(state_dict, model_path)
    print('models {} save successfully!'.format(model_path))

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0.1, mode='fan_in')
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0.1, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def visualize_sequences(batch, seq_len, return_fig=True):
    """
    visualize a sequence (imgs or flows)
    """
    sequences = []
    channels_per_frame = batch.shape[-1] // seq_len
    for i in range(batch.shape[0]):
        cur_sample = batch[i]  # [H,W,channels_per_frame * seq_len]
        if channels_per_frame == 2:
            sequence = [flow2img(cur_sample[:, :, j * channels_per_frame:(j + 1) * channels_per_frame])
                        for j in range(seq_len)]
        else:
            # BGR to RGB
            sequence = [cur_sample[:, :, j * channels_per_frame:(j + 1) * channels_per_frame][:, :, ::-1]
                        for j in range(seq_len)]
        sequences.append(np.hstack(sequence))
    sequences = np.vstack(sequences)

    if return_fig:
        fig = plt.figure()
        plt.imshow(sequences)
        return fig
    else:
        return sequences
    
def flow2img(flow_data):
    """
    convert optical flow into color image
    :param flow_data:  shape [H,W,2]
    :return: color image
    """
    # print(flow_data.shape)
    # print(type(flow_data))
    u = flow_data[:, :, 0]
    v = flow_data[:, :, 1]

    UNKNOW_FLOW_THRESHOLD = 1e7
    pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    idx_unknown = (pr1 | pr2)
    u[idx_unknown] = v[idx_unknown] = 0

    # get max value in each direction
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u / maxrad + np.finfo(float).eps
    v = v / maxrad + np.finfo(float).eps

    img = compute_color(u, v)

    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: horizontal optical flow
    :param v: vertical optical flow
    :return:
    """

    height, width = u.shape
    img = np.zeros((height, width, 3))

    NAN_idx = np.isnan(u) | np.isnan(v)
    u[NAN_idx] = v[NAN_idx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

    return img

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel