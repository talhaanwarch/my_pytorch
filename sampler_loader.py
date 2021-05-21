# -*- coding: utf-8 -*-
"""
Created on Fri May 21 09:21:10 2021

@author: TAC
"""

import torch 
from torchvision import transforms 
import torchvision.models as models
import torch.nn as nn
from glob import glob
import re
from sklearn.utils import resample
import numpy as np
#first list all folders and thier labels
_pos_path=[folder for folder in glob('covid/*/')]
_neg_path=[folder for folder in glob('non-covid/*/')]
_pos_label=[0]*len(_pos_path)
_neg_label=[1]*len(_neg_path)

path=_pos_path+_neg_path
label=_pos_label+_neg_label

from PIL import Image
class dfloader(torch.utils.data.Dataset):
    
    def __init__(self,path,label,n_samples=100,transform=None):
        self.path = path    
        self.label=label
        self.transform=transform
        self.n_samples=n_samples
        
    def __getitem__(self,index):
        if type(index) == torch.Tensor:
          index = index.item()
        #choose a path and correponding label
        path_ind=self.path[index]
        label_ind=self.label[index]
        print('path_ind',path_ind)
        #now open that path folder
        img_path=glob(path_ind+'*.jpg')
        #find all image ids
        img_idx=[int(re.findall(r'\d+', i.split('\\')[-1])[0]) for i in img_path]    
        #sort all path accordinh to sequence 
        img_idx, img_path = zip(*sorted(zip(img_idx, img_path)))
        #now resample the img_idx
        if len(img_idx)>self.n_samples:
            img_idx=sorted(resample(img_idx,replace=False,n_samples=self.n_samples))
        else:
            img_idx=sorted(resample(img_idx,replace=True,n_samples=self.n_samples))
        #now find correponding image_path
        img_path=[img_path[i] for i in img_idx]
        
        #now loop all image_path
        #img_list = [Image.open(p).convert('L') for p in img_path]
        img_list=[]
        for p in img_path:
            img=Image.open(p).convert('L')
            if self.transform:
                img_list.append(self.transform(img))
        
        #label_list = [label_ind]*len(img_list)
      
        
        #convert to array
        image_array=np.stack((img_list))
        image_array=np.moveaxis(image_array,0,3)
        #label_array=np.array(label_list)
        
        return image_array ,label_ind
    
    def __len__(self):
        return len(self.path)
from PIL import Image


aug=transforms.Compose([
    #transforms.Resize((224,224)), a;ready resized
    #transforms.Grayscale(),#in code gray scales
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    
    transforms.ToTensor(),
    transforms.Normalize([0.4, ], [0.3,]),
                        ])

# def my_collate(batch):
#     #move axis
#     data=np.concatenate([np.moveaxis(i[0],0,3) for i in batch])
#     label=np.array([i[1] for i in batch])
#     return torch.tensor(data,dtype=torch.float),torch.tensor(label)
    
from torch.utils.data.dataloader import DataLoader
train=dfloader(path,label,transform=aug)
train_loader = DataLoader(train,shuffle=True,num_workers=0,batch_size=3)

for i,j in train_loader:
    print(i.shape,j)
    break
    

