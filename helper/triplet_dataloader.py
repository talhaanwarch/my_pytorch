# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 16:12:38 2020

@author: TAC
"""

from PIL import Image
import random

class SiameseNetworkDataset():
    
    def __init__(self,imageFolderDataset,image_D='3D',transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform=transform
        self.image_D=image_D
        
    def __getitem__(self,index):
        anchor_img,anchor_class = random.choice(self.imageFolderDataset.imgs)
        #select the image from same class as of anchor class
        while True:
            positive_img,positive_class = random.choice(self.imageFolderDataset.imgs)
            if positive_class==anchor_class:
                break
            
        #select the negative image from different class as of anchor class
        while True:
            neagtive_img,negative_class = random.choice(self.imageFolderDataset.imgs)
            if negative_class!=anchor_class:
                break
            
        #now load image from path
        anchor_img=Image.open(anchor_img)
        positive_img=Image.open(positive_img)
        neagtive_img=Image.open(neagtive_img)
        
        if self.image_D=='2D':
            anchor_img=anchor_img.convert("L")
            positive_img=positive_img.convert("L")
            neagtive_img=neagtive_img.convert("L")
        
        if self.transform:
           anchor_img=self.transform(anchor_img) 
           positive_img=self.transform(positive_img) 
           neagtive_img=self.transform(neagtive_img) 

        
        return anchor_img, positive_img ,neagtive_img
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
