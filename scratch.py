from torch.utils.data import DataLoader
import torch
import tqdm
from dataset.YUVvideo_dataset import VideoDataset
import os

data_path = '/home/Dataset/UCSD_ped/UCSD_ped2'
trainset_yuvroot=os.path.join(data_path, "train_recyuv400/")
testset_yuvroot=os.path.join(data_path,  "test_recyuv400/")
trainset_mvroot=os.path.join(data_path,  "trainmv_txt/")
testset_mvroot=os.path.join(data_path,  "testmv_txt/")

dataset = VideoDataset(1, 3,
                        trainset_yuvroot, trainset_mvroot)
dataloader = DataLoader(dataset=dataset, batch_size=32, 
                        num_workers=8, shuffle=True)

# test_dataset = VideoDataset(1, 3,
#                             testset_yuvroot, testset_mvroot)
# test_loader = DataLoader(dataset=test_dataset, batch_size=32, 
#                             num_workers=8, shuffle=False, drop_last=True)

for epoch in range(0, 10):
    for batch, train_data in tqdm(enumerate(dataloader),desc="Training Epoch %d" % (epoch + 1),total=len(dataloader)):
        sample_frames, sample_mvs, _, _, mv_last = train_data
        print(sample_frames.shape)
        print(sample_mvs.shape)