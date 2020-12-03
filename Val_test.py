# -*- coding: utf-8 -*-
import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset

import Unet_TriplePlus
from metrics import dice_coef, batch_iou, mean_iou, iou_score ,ppv,sensitivity
import losses
from utils import str2bool, count_params
from sklearn.externals import joblib
import SimpleITK as sitk
import imageio
#import ttach as tta

wt_dices = []
tc_dices = []
et_dices = []
wt_sensitivities = []
tc_sensitivities = []
et_sensitivities = []
wt_ppvs = []
tc_ppvs = []
et_ppvs = []
wt_Hausdorf = []
tc_Hausdorf = []
et_Hausdorf = []

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')
    parser.add_argument('--mode', default=None,
                        help='')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='DeepResUNet')
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--dataset', default="jiu0Monkey",
                        help='dataset name')
    parser.add_argument('--input-channels', default=4, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss')
    parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=10, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--inference',type=bool,default=False)                        

    args = parser.parse_args()

    return args

#获取某个分块的位置信息（0 32 64 96 128）以及 该块属于哪个病例
def GetPatchPosition(PatchPath):
    npName = os.path.basename(PatchPath)
    firstName = npName
    overNum = npName.find(".npy")
    npName = npName[0:overNum]
    PeopleName = npName
    overNum = npName.find("_")
    while(overNum != -1):
        npName = npName[overNum+1:len(npName)]
        overNum = npName.find("_")
    overNum = firstName.find("_"+npName+".npy")
    PeopleName = PeopleName[0:overNum]
    return int(npName),PeopleName

def sort_key(s):
    if s:
        try:
            npName = os.path.basename(s)
            firstName = npName
            overNum = npName.find(".npy")
            npName = npName[0:overNum]
            PeopleName = npName
            overNum = npName.find("_")
            while(overNum != -1):
                npName = npName[overNum+1:len(npName)]
                overNum = npName.find("_")
            c = npName
        except:
            c = -1
        return int(c)

def main():
    val_args = parse_args()

    #args = joblib.load('models/%s/args.pkl'%val_args.name)
    if not os.path.exists('output/%s'%val_args.name):
        os.makedirs('output/%s'%val_args.name)
    print('Config ----')
    for arg in vars(val_args):
        print('%s: %s'%(arg,getattr(val_args,arg)))
    print('-----------')

    joblib.dump(val_args,'models/%s/args.pkl'%val_args.name)

    #create model 
    print('=> creating model %s'%val_args.arch)
    model = Unet_TriplePlus.__dict__[val_args.arch](val_args)

    model = model.cuda()
    print(count_params(model))
    model.load_state_dict(torch.load('models/%s/model.pth'%val_args.name))
    model.eval() 
    #model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='max')

    #Data path 
    data_path = r'E:\wuyujie\BraTS2018_Challenge\IResUnet3P_3D\BraTs2018_3DVal_Data_Npy_32step'
    save_path = r'E:\wuyujie\BraTS2018_Challenge\IResUnet3P_3D\BraTs18_Val_output_0802'
    label_suf = '.nii.gz'

    savedir = 'output/%s/'%val_args.name
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        with torch.no_grad():
            startFlag = 1
            person_list = os.listdir(data_path)
            for person_name in person_list:
                person_path = os.path.join(data_path,person_name)
                img_list = os.listdir(person_path)
                img_list = sorted(img_list,key=sort_key)
                # 创建一个全黑的三维矩阵，用于存放还原的mask
                save_label = np.zeros([155,240,240],dtype=np.uint16)
                # 创建三个全黑的三维矩阵，分别用于预测出来的WT、TC、ET分块的拼接
                OneWT = np.zeros([160, 160, 160], dtype=np.uint16)
                OneTC = np.zeros([160, 160, 160], dtype=np.uint16)
                OneET = np.zeros([160, 160, 160], dtype=np.uint16)
                # 创建一个全黑的三维矩阵，用于存放一个人的预测
                OnePeople = np.zeros([160,160,160],dtype=np.uint16)
                for n_patch,img_name in enumerate(img_list):
                    #print(n_patch,img_name)
                    img_path = os.path.join(person_path,img_name)
                    img = np.load(img_path)
                    img = img.transpose((3,0,1,2))
                    img = img.astype("float32")
                    img = torch.from_numpy(img).unsqueeze(0)
                    img = img.cuda()
                    output = model(img)
                    output = torch.sigmoid(output).data.cpu().numpy()
                    PatchPosition, NameNow = GetPatchPosition(img_list[n_patch])
                    print(PatchPosition,NameNow)
                    # 预测分块的拼接
                    for i in range(output.shape[0]):
                        for idz in range(output.shape[2]):
                            for idx in range(output.shape[3]):
                                for idy in range(output.shape[4]):
                                    if output[i, 0,idz,idx,idy] > 0.5:      # WT拼接
                                        OneWT[PatchPosition + idz, idx, idy] = 1
                                    if output[i, 1, idz, idx, idy] > 0.5:  # TC拼接
                                        OneTC[PatchPosition + idz, idx, idy] = 1
                                    if output[i, 2, idz, idx, idy] > 0.5:  # ET拼接
                                        OneET[PatchPosition + idz, idx, idy] = 1
                #预测Mask的拼接、保存
                # OnePeople 0 1 2 4 => 增加或减少切片使得尺寸回到（155，240，240） => NII
                for idz in range(OneWT.shape[0]):
                    for idx in range(OneWT.shape[1]):
                        for idy in range(OneWT.shape[2]):
                            if (OneWT[idz, idx, idy] == 1):
                                OnePeople[idz, idx, idy] = 2
                            if (OneTC[idz, idx, idy] == 1):
                                OnePeople[idz, idx, idy] = 1
                            if (OneET[idz, idx, idy] == 1):
                                OnePeople[idz, idx, idy] = 4
                save_label[:,40:200,40:200] = OnePeople[3:158,:,:]
                saveout = sitk.GetImageFromArray(save_label)
                sitk.WriteImage(saveout,save_path  + "\\" + person_name + label_suf)
                
if __name__ == '__main__':
    main( )