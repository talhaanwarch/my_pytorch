# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 12:18:02 2020

@author: TAC
"""


import torch.nn as nn
class siamese_vgg16(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1=nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,stride=1,padding=1)
        self.conv1_2=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=1,padding=1)
        
        self.conv2_1=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,stride=1,padding=1)
        self.conv2_2=nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,stride=1,padding=1)
        
        self.conv3_1=nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,stride=1,padding=1)
        self.conv3_3=nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,stride=1,padding=1)
        
        self.conv4_1=nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,stride=1,padding=1)
        
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.flatten=nn.Flatten()
        self.fcLayer1=nn.Linear(in_features=18432,out_features=1024)
        self.fcLayer2=nn.Linear(in_features=1024,out_features=512)
        self.fcLayer3=nn.Linear(in_features=512,out_features=128)
        self.fcLayer4=nn.Linear(in_features=128,out_features=64)

        self.relu=nn.ReLU()
        
    def forward_once(self,x):
        x=self.relu(self.conv1_1(x))       
        x=self.relu(self.conv1_2(x))  
        x=self.maxpool(x)
        
        x=self.relu(self.conv2_1(x))  
        x=self.relu(self.conv2_2(x)) 
        x=self.maxpool(x)

        x=self.relu(self.conv3_1(x))  
        x=self.relu(self.conv3_1(x))  
        x=self.relu(self.conv3_3(x)) 
        x=self.maxpool(x)

        x=self.relu(self.conv4_1(x))  
        x=self.relu(self.conv4_1(x))  
        x=self.relu(self.conv4_1(x)) 
        x=self.relu(self.conv4_1(x)) 
        x=self.maxpool(x)
        
        
        x=x.view(x.size(0),-1)
        
        x=self.relu(self.fcLayer1(x))
        x=self.relu(self.fcLayer2(x))
        x=self.relu(self.fcLayer3(x))
        x=self.fcLayer4(x)
        return x    
    
    def forward(self, anchor, positive, negative):
        anchor = self.forward_once(anchor)
        positive = self.forward_once(positive)
        negative = self.forward_once(negative)
        return anchor, positive,negative
