from __future__ import print_function, absolute_import
import torch
from torch.utils.data import Dataset
import os, os.path
from torchvision import transforms
import numpy as np
import pdb
import cv2
import torch.nn as nn
import imageio


def img_tensor2numpy(img):
    # mutual transformation between ndarray-like imgs and Tensor-like images
    # both intensity and rgb images are represented by 3-dim data
    if isinstance(img, np.ndarray):
        return torch.from_numpy(np.transpose(img, [2, 0, 1]))
    else:
        return np.transpose(img, [1, 2, 0]).numpy()


def img_batch_tensor2numpy(img_batch):
    # both intensity and rgb image batch are represented by 4-dim data
    if isinstance(img_batch, np.ndarray):
        if len(img_batch.shape) == 4:
            return torch.from_numpy(np.transpose(img_batch, [0, 3, 1, 2]))
        else:
            return torch.from_numpy(np.transpose(img_batch, [0, 1, 4, 2, 3]))
    else:
        if len(img_batch.numpy().shape) == 4:
            return np.transpose(img_batch, [0, 2, 3, 1]).numpy()
        else:
            return np.transpose(img_batch, [0, 1, 3, 4, 2]).numpy()
        
def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)

###读取yuv的各分量
def readYuvFile(filename,width,height,frame_num,ImgChnNum) :
    fp=open(filename,'rb')
    uv_width=width//2
    uv_height=height//2
    Y=np.zeros((frame_num,height,width) ,np.uint8, 'C')
    U=np.zeros((frame_num,uv_height,uv_width),np.uint8,'C')
    V=np.zeros((frame_num,uv_height,uv_width) ,np.uint8,'C')
    U_expand=np.zeros((frame_num,height,width) ,np.uint8, 'C')
    V_expand=np.zeros((frame_num,height,width) ,np.uint8, 'C')
    for k in range(frame_num):
        for m in range(height):
            for n in range(width):
                Y[k,m,n]=ord(fp.read(1))
        if ImgChnNum==3:
            for m in range(uv_height) :
                for n in range(uv_width) :
                    U[k,m,n]=ord(fp.read(1))
                    V[k,m,n]=ord(fp.read(1)) 
    if ImgChnNum==3:
        U_expand[:,0:256:2,0:256:2]=U
        U_expand[:,0:256:2,1:256:2]=U
        U_expand[:,1:256:2,0:256:2]=U
        U_expand[:,1:256:2,1:256:2]=U
        V_expand[:,0:256:2,0:256:2]=V
        V_expand[:,0:256:2,1:256:2]=V
        V_expand[:,1:256:2,0:256:2]=V
        V_expand[:,1:256:2,1:256:2]=V
        fp.close()
        return (Y,U_expand,V_expand)
    fp.close()
    return (Y,U,V)

def readmv(mv_root,x_txt,y_txt):
    path1=mv_root+x_txt
    path2=mv_root+y_txt
    file1=open(path1,'r')
    file2=open(path2,'r')
    mv=np.zeros((64,64,2))
    content1=file1.readlines()
    content2=file2.readlines()
    for i in range(64):
        content1[i] = content1[i].replace('\n', '')
        content2[i] = content2[i].replace('\n', '')
        contentx = content1[i].split(',')
        contenty = content2[i].split(',')
        for j in range(64):
            mv[i][j][0] = int(contentx[j])
            mv[i][j][1] = int(contenty[j])
    file2.close()  #文件打开，使用完毕后需要关闭
    file2.close()
    return mv

def change_shape(y):
    x = np.transpose(y, [2, 3, 0, 1])  # [#t x cx h x w ] to [h,w,t,c]
    x = np.reshape(x, (x.shape[0], x.shape[1], -1))#[h,w,#frame,c] to [h,w,t*c]
    x = np.transpose(x, [2, 0, 1]) #[h,w,t*c] to [t*c,h,w]
    return x


# N  x (T*C)x H x W
class VideoDataset(Dataset):
    def __init__(self,ImgChnNum,video_root,
                clip_length=4,
                interval=4,
                time_step=4,
                use_cuda=False, transform=None):
        self.ImgChnNum=ImgChnNum
        self.clip_length=clip_length+1
        self.interval=interval
        self.time_step=time_step
        # if self.ImgChnNum==3:
        #     self._input_mean = torch.from_numpy(
        #         np.array([0.475, 0.497, 0.497]).reshape((1, 3, 1, 1))).float()
        #     self._input_std = torch.from_numpy(
        #         np.array([0.188, 0.017, 0.017]).reshape((1, 3, 1, 1))).float()  # 进行SH数据集的均值标准差设定
        if self.ImgChnNum==3:
            self._input_mean = torch.from_numpy(
                np.array([0.408, 0.502, 0.502]).reshape((1, 3, 1, 1))).float()
            self._input_std = torch.from_numpy(
                np.array([0.248, 0.028, 0.028]).reshape((1, 3, 1, 1))).float()
        elif self.ImgChnNum==1:
            self._input_mean=torch.from_numpy(np.array([0.446])).float()
            self._input_std=torch.from_numpy(np.array([0.179])).float()  # 进行UCSD数据集的均值标准差设定
        # dir name
        self.video_root = video_root
        # video_name_list, video names
        self.video_list = [name for name in os.listdir(self.video_root)]
        self.video_list.sort()

        self.video_clip_list=[]  # 存放的GOP的I帧的idx
        self.video_yuv=[]##每个视频的名字和yuv分量
        width,height=256,256
        
        for k in range(len( self.video_list)):
            video_name=self.video_list[k]
            print("%s is load"%video_name)
            v_path = self.video_root +video_name  #01.yuv
            
            file_size = os.path.getsize(v_path)
            if self.ImgChnNum==1:
                frame_num=file_size // (width*height)###yuv400每1个像素对应1个值  计算帧数
            else:
                frame_num=file_size // (width*height*3 // 2)###yuv420每4个像素对应6个值

            #print(frame_num)
            
            #####读取I帧数据
            yuv_data=readYuvFile(v_path,width,height,frame_num,self.ImgChnNum)
            
            self.video_yuv.append([video_name,yuv_data])
            if frame_num<self.clip_length*self.interval:  # 5*4=20 5个GOP，4个GOP的Iframe去预测第五个GOP的Iframe  如果视频小于5*GOP则报错
                print("The video %s have no enough frames" %video_name)
                continue
            else:
                frame_begain_idx=1
                while True:
                    frame_end_idx=frame_begain_idx+((self.clip_length-1)*self.interval)  # 1+4*4=17
                    if frame_end_idx>frame_num:
                        break
                    if frame_num-frame_end_idx>=3:  #make sure the num of P frame in the last GOP >= sampled_mv_num
                        frame_idx=[k for k in range(frame_begain_idx, frame_end_idx+1, self.interval)] # 1,5,9,13,17——>5,9,13,17,21
                        #print(frame_idx)
                        self.video_clip_list.append([video_name,frame_idx])  # 把5个GOP当做一个video clip 存入
                        frame_begain_idx=frame_begain_idx+self.time_step  # 以5个GOP为滑动窗口大小，每次滑动一个GOP 1——>5——>9
                    else:
                        break         
        self.idx_num = len(self.video_clip_list)
        print("%d clips load" %self.idx_num)  # 一个clip是5个GOP
        self.use_cuda = use_cuda
        self.transform = transform
        self.num_predicted_frame=1##预测帧的个数
        #exit()

    def __len__(self):
        return self.idx_num

    def __getitem__(self, item):
        
        """ get a video clip with stacked frames indexed by the (idx) """
        clip_path = self.video_clip_list[item] # idx file path  [video_name,frame_idx]
        v_name = clip_path[0]  # video name
        frame_idx = clip_path[1]  # frame index list for a video clip
        # pred_frame_test=(np.array(frame_idx[-self.num_predicted_frame:])-1).tolist()#预测帧的序号，frame_idx的最后一个
        pred_frame_test=frame_idx[-1]-1 #预测帧的序号，frame_idx的最后一个
        #print(pred_frame_test)
        
        for k in range(len( self.video_list)):###此video所对应的yuv
            if self.video_yuv[k][0]==v_name:
                yuv=self.video_yuv[k][1]
    
        ##I帧  取出5个GOP的
        #pdb.set_trace()
        frames = []
        for k in range(len(frame_idx)):
            #pdb.set_trace()
            frame_index=frame_idx[k]-1
            if self.ImgChnNum==1:
                img=(yuv[0][frame_index]-16.0) / (235.0-16.0) ##归一化，Y范围是16-235
                img=np.expand_dims(img,axis=2)##将维度从256*256扩充到256*256*1
            elif self.ImgChnNum==3:
                img = np.zeros((256, 256, 3))
                img[:,:,0]=(yuv[0][frame_index]-16.0) / (235.0-16.0) ##归一化，Y范围是16-235
                img[:,:,1]=(yuv[1][frame_index]-16.0) / (240.0-16.0) ##归一化，U范围是16-240
                img[:,:,2]=(yuv[2][frame_index]-16.0) / (240.0-16.0) ##归一化，V范围是16-240
                
            if img is None:
                print('Error: loading video %s failed.' % v_name)
                img = np.zeros((256, 256, 3))
            #img = img[..., ::-1]##通道BGR to RGB
            frames.append(img)
        
        frames = np.array(frames)#T x H x W x C
        frames = np.transpose(frames, (0, 3, 1, 2))#T x Cx H x W 
        output_frames = torch.from_numpy(frames).float()
        output_frames = (output_frames - self._input_mean) / self._input_std
        #T x Cx H x W to [t*c,h,w]
        output_frames=change_shape(output_frames)
        
        return output_frames,pred_frame_test,v_name  # 返回5个GOP所有的I，预测帧的序号，视频名称