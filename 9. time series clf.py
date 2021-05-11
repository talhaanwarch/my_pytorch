# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:44:12 2021

@author: TAC
"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
#first create a dataframe with file path and labels
from glob import glob
import pandas as pd
ctl_folder_path='D:/Datasets/EEG dataset/Pain Machine Learning Project/A/Control/Cold Pressor'
ctl_file_path=[i for i in glob(ctl_folder_path+'/*.cnt')]
ctl_label=[0]*len(ctl_file_path)
ctl_df=pd.DataFrame(zip(ctl_file_path,ctl_label),columns=['path','label'])


pat_folder_path='D:/Datasets/EEG dataset/Pain Machine Learning Project/A/Patient/Cold Pressor'
pat_file_path=[i for i in glob(pat_folder_path+'/*.cnt')]
pat_label=[1]*len(pat_file_path)
pat_df=pd.DataFrame(zip(pat_file_path,pat_label),columns=['path','label'])

df=pd.concat([ctl_df,pat_df])

#write a data loader to load files from disk in batches
import torch
from torch.utils.data.dataloader import DataLoader
import mne
import numpy as np

class DFloader(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, df):
        'Initialization'
        self.df = df
        self.fs=1000
        self.sec=1
        self.channels=61
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file = self.df.iloc[index,0]
        label=self.df.iloc[index,1]

        data=mne.io.read_raw_cnt(file).get_data()
        seg=data.shape[1]%(self.fs*self.sec)
        data=data[0:self.channels,seg::]
        data=data.reshape(self.channels,-1,self.fs*self.sec)
        data=np.swapaxes(data,0,1)
        return data,label

#custom collate function to convert trials to batch
def my_collate(batch):
    data=np.concatenate([i[0] for i in batch])    
    label=np.concatenate([[i[1]]*len(i[0]) for i in batch]) #multiply each label with corresponding trials
    return torch.tensor(data,dtype=torch.float),torch.tensor(label)

# #for test purpose if every thing working or not    
# train=DFloader(df)    
# train_loader = DataLoader(train,shuffle=True,num_workers=0,batch_size=4,collate_fn=my_collate)    

# for file,label in train_loader:
#     print(len(file),len(label))    
#     break

#create CNN model
import torch.nn as nn 
class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1=nn.Conv1d(in_channels=61, out_channels=5, kernel_size=3,stride=1,padding=0)#padding 0: defualt keras
        self.conv_2=nn.Conv1d(in_channels=5, out_channels=5, kernel_size=3,stride=1,padding=0)
        self.batch_norm=nn.BatchNorm1d(5)
        self.leaky=nn.LeakyReLU()
        self.maxpool=nn.MaxPool1d(kernel_size=2,stride=2)
        self.avgpool=nn.AvgPool1d(kernel_size=2,stride=2)
        self.dropout=nn.Dropout2d(0.5)
        self.globalavg=nn.AdaptiveAvgPool1d(5)
        self.flat=nn.Flatten()
        self.linear=nn.Linear(in_features=25,out_features=1)
        
    def forward(self,x):
        x=self.conv_1(x)
        x=self.batch_norm(x)
        x=self.leaky(x)
        x=self.maxpool(x)
        
        x=self.conv_2(x)
        x=self.leaky(x)
        x=self.maxpool(x)
        x=self.dropout(x)
        
        for i in range(2):
            x=self.conv_2(x)
            x=self.leaky(x)
            x=self.avgpool(x)
            x=self.dropout(x)
        x=self.globalavg(x)
        x=self.flat(x)
        x=self.linear(x)
        return x
device='cuda'  #for gpu      

#test model
import torch        
x=torch.randn(5,61,1000)
net=DeepCNN()
out=net(x)
print(out.shape)
    

# #torch summary
# from torchsummary import summary
# model = DeepCNN().to(device)
# summary(model, (61, 1000))

#split data to train test and load it
from sklearn.model_selection import train_test_split
train,test=train_test_split(df, test_size=0.2)    

train=DFloader(train)    
train_loader = DataLoader(train,shuffle=True,num_workers=0,batch_size=6,collate_fn=my_collate)     
    
test=DFloader(test)    
test_loader = DataLoader(test,shuffle=True,num_workers=0,batch_size=6,collate_fn=my_collate)

#initialize model and its parameters
model=   DeepCNN().to(device)
opt=torch.optim.Adam(params=model.parameters(),lr=0.0001)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode ='min',factor=0.5, patience=4, verbose=True)

# for file,label in train_loader:
#     file,label = file.to(device),label.to(device)  
#     print(len(file),len(label))
      
def get_accuracy(y_true, y_pred):
    assert y_true.ndim == 1 and y_true.size() == y_pred.size()
    y_pred = y_pred > 0.5
    return (y_true == y_pred).sum().item() / y_true.size(0)

def fit_train(loader):
    loss_sum=0
    acc_sum=0
    for batch in loader:
        img,label=batch
        img,label = img.to(device),label.to(device)
        
        out=model(img)
       # print('img',img.shape,'label',label.shape,'target',out.shape)
        loss=criterion(out.view(-1),label.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum+=loss.item()
        acc_sum += get_accuracy(label,out.view(-1))
    return loss_sum,acc_sum
def fit_val(loader):
    loss_sum=0
    acc_sum=0
    for batch in loader:
        img,label=batch
        img,label = img.to(device),label.to(device)
        out=model(img)
        loss=criterion(out.view(-1),label.float())
        loss_sum+=loss.item()
        acc_sum += get_accuracy(label,out.view(-1))
    return loss_sum,acc_sum    
    

from time import time
def fit(train_loader,val_loader,model,epoch=10,scheduler_step=None,verbose=None):
  
  train_loss_plt=[]
  val_loss_plt=[]
  train_acc_plt=[]
  val_acc_plt=[]
  for ep in range(epoch):
    start=time()
    #start training loop
    train_loss,train_acc=fit_train(train_loader)
    #start validation loop
    model.eval()
    with torch.no_grad():
        val_loss,val_acc=fit_val(val_loader)
    end=np.round((time()-start)/60,2) #time in minute
    model.train()

    #calculate print and append the results for plotting purpose
    val_avg_loss=np.round(val_loss/len(val_loader),2)#val loss of all batches of one epoch
    train_avg_loss=np.round(train_loss/len(train_loader),2)# train loss of all batches of one epoch
    train_avg_acc=np.round(train_acc/len(train_loader),2)#train acc of all batches of one epoch
    val_avg_acc=np.round(val_acc/len(val_loader),2)#val acc of all batches of one epoch
    if scheduler_step:
      scheduler.step(val_avg_loss)
    if verbose:
      print('Epoch {}, time {} , train acc  {}, train loss {} , val acc is {}, loss is {}, learning rate is {} '.format
            (ep,end,train_avg_acc,train_avg_loss,val_avg_acc,val_avg_loss,opt.param_groups[0]['lr']))
    train_loss_plt.append(train_avg_loss)  #append loss of training data  
    val_loss_plt.append(val_avg_loss)     #append loss of validation data
    train_acc_plt.append(train_avg_acc)  #append acc of training data  
    val_acc_plt.append(val_avg_acc)     #append acc of validation data
  
  return [train_loss_plt,val_loss_plt,train_acc_plt,val_acc_plt]  
    
res=fit(train_loader,test_loader,model=model,epoch=15,scheduler_step=True,verbose=True)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    