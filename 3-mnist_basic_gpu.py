# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:14:18 2020

@author: TAC
"""


import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device(type='cpu')

device = get_default_device()
print('default device is',device)


train=torchvision.datasets.MNIST(root='mnist_data/',train=True,transform=torchvision.transforms.ToTensor(),download=False)
test=torchvision.datasets.MNIST(root='mnist_data/',train=False,transform=torchvision.transforms.ToTensor(),download=False)

tds,vds=torch.utils.data.random_split(train, lengths=[50000,10000])
print(len(tds),len(vds))

batch_size=128
train_loader=torch.utils.data.DataLoader(tds,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=False)
val_loader=torch.utils.data.DataLoader(vds,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=False)


for img, _ in train_loader:
    print(img[0].dtype)
    plt.imshow(torchvision.utils.make_grid(img,nrow=8).permute(1,2,0))
    break


def accuracy(pred,actual):
    max_prob,pred=torch.max(pred,dim=1)
    acc=torch.true_divide(torch.sum(pred==actual), len(pred))
    return acc

class mnist_cnn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fcLayer1=torch.nn.Linear(in_features=784,out_features=100)
        self.fcLayer2=torch.nn.Linear(in_features=100,out_features=10)
        self.relu=torch.nn.ReLU()
        self.softmax=torch.nn.Softmax(dim=1)
        
    def forward(self,x):
        x=x.reshape(-1,784)
        x=self.fcLayer1(x)
        x=self.relu(x)
        x=self.fcLayer2(x)
        x=self.softmax(x)
        return x    

    def training_step(self,batch):
        images,labels=batch
        
        images,labels=images.to(device),labels.to(device)

        y_pred=self(images)
        loss=torch.nn.functional.cross_entropy(y_pred,labels)
        return loss
    def validation_step(self,batch):
        images,labels=batch
        
        images,labels=images.to(device),labels.to(device)
        
        y_pred=self(images)
        loss=torch.nn.functional.cross_entropy(y_pred, labels)
        acc=accuracy(pred=y_pred, actual=labels)
        return {'val_loss':loss,'val_acc':acc}
    
#    def validation_result_at_epoch_end(self,val_acc):
#        batch_losss=[x['val_']]
        
model=   mnist_cnn().to(device)
opt=torch.optim.Adam(params=model.parameters(),lr=0.0001)

from time import time
start=time()

ep_acc=[]
ep_loss=[]
for ep in range(5):
    for tr_batch in train_loader:
        loss=model.training_step(tr_batch)
        loss.backward()
        opt.step()
        opt.zero_grad
    print('epoch {} training loss is {}'.format(ep,loss))
        
    val_loss_sum=torch.tensor(0.)
    val_acc_sum=torch.tensor(0.)
    for val_batch in val_loader:
        out=model.validation_step(val_batch)
        val_loss_sum+=out['val_loss']
        val_acc_sum+=out['val_acc']
    b=len(val_loader)
    print('epoch {} val loss is {} and val acc  is {} '.format(ep,val_loss_sum.item()/b,val_acc_sum.item()/b))
    ep_loss.append(val_loss_sum/b)
    ep_acc.append(val_acc_sum/b)
print('time taken :', time()-start) 

plt.plot(ep_loss)
plt.plot(ep_acc)




