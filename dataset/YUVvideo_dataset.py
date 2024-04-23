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
    file2.close()  
    file2.close()
    return mv

def change_shape(y):
    x = np.transpose(y, [2, 3, 0, 1])  # [#t x cx h x w ] to [h,w,t,c]
    x = np.reshape(x, (x.shape[0], x.shape[1], -1))#[h,w,#frame,c] to [h,w,t*c]
    x = np.transpose(x, [2, 0, 1]) #[h,w,t*c] to [t*c,h,w]
    return x


# N  x (T*C)x H x W
class VideoDataset(Dataset):
    def __init__(self,ImgChnNum,sampled_mv_num,video_root, mv_root,
                last_mv=False,
                clip_length=4,
                interval=4,
                time_step=4,
                use_cuda=False, transform=None):
        self.ImgChnNum=ImgChnNum
        self.sampled_mv_num=sampled_mv_num
        self.last_mv=last_mv
        self.clip_length=clip_length+1
        self.interval=interval
        self.time_step=time_step
        if self.ImgChnNum==3:
            self._input_mean = torch.from_numpy(
                np.array([0.408, 0.502, 0.502]).reshape((1, 3, 1, 1))).float()
            self._input_std = torch.from_numpy(
                np.array([0.248, 0.028, 0.028]).reshape((1, 3, 1, 1))).float()  
        elif self.ImgChnNum==1:
            self._input_mean=torch.from_numpy(np.array([0.446])).float()
            self._input_std=torch.from_numpy(np.array([0.179])).float() 
        # dir name
        self.video_root = video_root
        self.mv_root=mv_root
        # video_name_list, video names
        self.video_list = [name for name in os.listdir(self.video_root)]
        self.video_list.sort()

        self.video_clip_list=[]  
        self.video_yuv=[]
        width,height=256,256
        
        for k in range(len( self.video_list)):
            video_name=self.video_list[k]
            print("%s is load"%video_name)
            v_path = self.video_root +video_name  #01.yuv
            
            file_size = os.path.getsize(v_path)
            if self.ImgChnNum==1:
                frame_num=file_size // (width*height)
            else:
                frame_num=file_size // (width*height*3 // 2)

            #print(frame_num)
            MV_data=[]
            for k in range(frame_num):
                #print(k)
                x_txt=video_name[0:-4]+'/'+str(k).zfill(5)+'_x.txt'
                y_txt=video_name[0:-4]+'/'+str(k).zfill(5)+'_y.txt'
                MV=readmv(self.mv_root,x_txt,y_txt)
                MV_data.append(MV) 
            
            #####读取I帧数据
            yuv_data=readYuvFile(v_path,width,height,frame_num,self.ImgChnNum)
            
            self.video_yuv.append([video_name,yuv_data,MV_data])
            if frame_num<self.clip_length*self.interval:
                print("The video %s have no enough frames" %video_name)
                continue
            else:
                frame_begain_idx=1
                while True:
                    frame_end_idx=frame_begain_idx+((self.clip_length-1)*self.interval)  # 1+4*4=17
                    if frame_end_idx>frame_num:
                        break
                    if frame_num-frame_end_idx>=self.sampled_mv_num:  #make sure the num of P frame in the last GOP >= sampled_mv_num
                        frame_idx=[k for k in range(frame_begain_idx, frame_end_idx+1, self.interval)] # 1,5,9,13,17——>5,9,13,17,21
                        #print(frame_idx)
                        self.video_clip_list.append([video_name,frame_idx]) 
                        frame_begain_idx=frame_begain_idx+self.time_step  
                    else:
                        break         
        self.idx_num = len(self.video_clip_list)
        print("%d clips load" %self.idx_num)
        self.use_cuda = use_cuda
        self.transform = transform
        self.num_predicted_frame=1
        #exit()

    def __len__(self):
        return self.idx_num

    def __getitem__(self, item):
        
        """ get a video clip with stacked frames indexed by the (idx) """
        clip_path = self.video_clip_list[item] # idx file path  [video_name,frame_idx]
        v_name = clip_path[0]  # video name
        frame_idx = clip_path[1]  # frame index list for a video clip
        # pred_frame_test=(np.array(frame_idx[-self.num_predicted_frame:])-1).tolist()
        pred_frame_test=frame_idx[-1]-1 
        #print(pred_frame_test)
        
        for k in range(len( self.video_list)):
            if self.video_yuv[k][0]==v_name:
                yuv=self.video_yuv[k][1]
                MV_data=self.video_yuv[k][2]

        #pdb.set_trace()
        frames = []
        for k in range(len(frame_idx)):
            #pdb.set_trace()
            frame_index=frame_idx[k]-1
            if self.ImgChnNum==1:
                img=(yuv[0][frame_index]-16.0) / (235.0-16.0) 
                img=np.expand_dims(img,axis=2)
            elif self.ImgChnNum==3:
                img = np.zeros((256, 256, 3))
                img[:,:,0]=(yuv[0][frame_index]-16.0) / (235.0-16.0) 
                img[:,:,1]=(yuv[1][frame_index]-16.0) / (240.0-16.0) 
                img[:,:,2]=(yuv[2][frame_index]-16.0) / (240.0-16.0)
                
            if img is None:
                print('Error: loading video %s failed.' % v_name)
                img = np.zeros((256, 256, 3))

            frames.append(img)
        
        frames = np.array(frames)#T x H x W x C
        frames = np.transpose(frames, (0, 3, 1, 2))#T x Cx H x W 
        output_frames = torch.from_numpy(frames).float()
        output_frames = (output_frames - self._input_mean) / self._input_std
        #T x Cx H x W to [t*c,h,w]
        output_frames=change_shape(output_frames)

        # MV
        MVs = []  

        # MV stack in channel
        for k in range(len(frame_idx)):
            #pdb.set_trace()
            sample_mv= np.zeros((64, 64, 2*self.sampled_mv_num))           
            for m in range(self.sampled_mv_num):
                frame_index=frame_idx[k]+m 
                mv_img=MV_data[frame_index]
                #substract mean ,[-20,20]-->[0,255]
                img_x,img_y=np.split(mv_img,[1],axis=2)                               
                img_x=img_x-img_x.mean()
                img_y=img_y-img_y.mean()
                mv_img=np.dstack((img_x, img_y))
                mv_img = clip_and_scale(mv_img, 20)
                mv_img += 128
                mv_img = (np.minimum(np.maximum(mv_img, 0), 255)).astype(np.uint8)
                sample_mv[:,:,m*2] = mv_img[:,:,0]
                sample_mv[:,:,m*2+1] = mv_img[:,:,1]
            MVs.append(sample_mv)


        #pdb.set_trace()
        MVs = np.array(MVs)#T x H x W x C , C=c*sampled_mv_num 

        MV = MVs[:-1] if not self.last_mv else MVs[-1:]
        
        mv_last=MVs[-1:][0]
        mv_last_final=mv_last[:,:,0:2] 
        MV = np.transpose(MV, (0, 3, 1, 2))#T x Cx H x W 
        output_MV = torch.from_numpy(MV).float()/ 255.0 #array to tensor
        output_MV = (output_MV - 0.5)
        #T x Cx H x W to [t*c,h,w]
        output_MV=change_shape(output_MV)
        
        return output_frames,output_MV,pred_frame_test,v_name,mv_last_final 