# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 11:59:01 2020

@author: TAC
"""

import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader



data=pd.read_csv('data_file.csv',header=None)
data=data.sample(frac=1).reset_index(drop=True)


features=data.iloc[:,0:-1].values
labels=data.iloc[:,-1].values

features=torch.Tensor(features)
labels=torch.Tensor(labels).view(labels.shape[0],-1)

tds=TensorDataset(features,labels)

tdl=DataLoader(tds,batch_size=4)

#create model
def seq_model():
    model=torch.nn.Sequential(
        torch.nn.Linear(in_features=features.shape[1], out_features=1),
        torch.nn.Sigmoid()
        )   
    return model
#model=seq_model()


class nonseqV1(torch.nn.Module):
    def __init__(self):
        super(nonseqV1,self).__init__()
        self.fcLayer=torch.nn.Linear(in_features=features.shape[1], out_features=1)
        self.sigmoid=torch.nn.Sigmoid()
        self.relu=torch.nn.ReLU()
        
    def forward(self,x):
        x=self.fcLayer(x)
        x=self.sigmoid(x)
        return x
model=nonseqV1()



class nonseqV2(torch.nn.Module):
    def __init__(self):
        super(nonseqV2,self).__init__()
        self.fcLayer1=torch.nn.Linear(in_features=337, out_features=100) 
        #337 is the input dim, 100 is hidden layer neurons
        self.fcLayer2=torch.nn.Linear(in_features=100, out_features=1)
        #100 is the input neuron comping from previous hidden layer and 1 is output
        self.relu=torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
        
    def forward(self,x):
        x=self.fcLayer1(x)
        x=self.relu(x)
        x=self.fcLayer2(x)
        x=self.sigmoid(x)
        return x

model=nonseqV2()

bce=torch.nn.BCELoss()
opt=torch.optim.Adam(params=model.parameters(),lr=0.0001)

for epoch in range(10):
    for xb,yb in tdl:
        pred=model(xb)
        loss=bce(pred,yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    print(loss.item())
