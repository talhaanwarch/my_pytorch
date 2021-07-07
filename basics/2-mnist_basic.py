# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 13:18:57 2020

@author: TAC
"""


import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
#dataset=torchvision.datasets.MNIST(root='mnist_data/',download=True)

train=torchvision.datasets.MNIST(root='mnist_data/',train=True,transform=torchvision.transforms.ToTensor(),download=False)
print(len(train))

test=torchvision.datasets.MNIST(root='mnist_data/',train=False,transform=torchvision.transforms.ToTensor(),download=False)
print(len(test))



tds,vds=torch.utils.data.random_split(train, lengths=[50000,10000])
print(len(tds),len(vds))         


batch_size=256
train_loader=torch.utils.data.DataLoader(tds,batch_size=batch_size,shuffle=True)
val_loader=torch.utils.data.DataLoader(vds,batch_size=batch_size,shuffle=False)


class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fcLayer1=torch.nn.Linear(in_features=784, out_features=10)
        #784 is obtained by flattening the dimension of images (28*28)
        # 10 is the number of classes
        self.softmax=torch.nn.Softmax(dim=1)

    def forward(self,x):
        x=x.reshape(-1,784)
        x=self.fcLayer1(x) 
        x=self.softmax(x)
        return x
        
        
model=model()       

for images,labels in train_loader:
    y_pred=model(images)
    print(images.shape,labels.shape,y_pred.shape)
    break
        

def accuracy(pred,actual):
    max_prob,pred=torch.max(pred,dim=1)
    acc=torch.true_divide(torch.sum(pred==labels),len(pred))
    return acc
        
        
loss_func=torch.nn.CrossEntropyLoss() 
opt=torch.optim.Adam(params=model.parameters(),lr=0.0001)
   
for ep in (range(5)):    
    for images,labels in train_loader:
        y_pred=model(images)
        acc=accuracy(y_pred,labels)
        loss=loss_func(y_pred,labels)
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    val_loss_sum=torch.tensor(0.)
    val_acc_sum=torch.tensor(0.)
    for b,(val_img,val_labels) in enumerate(val_loader):
        val_pred=model(val_img)
        val_loss=loss_func(val_pred,val_labels)
        val_loss_sum+=val_loss
        #val_acc=accuracy(val_pred,val_labels)
        max_prob,val_pred=torch.max(val_pred,dim=1)
        val_acc=torch.true_divide(torch.sum(val_pred==val_labels),len(val_pred))
        val_acc_sum+=val_acc
    print('After Epoch {} val loss is {} and val acc is {}'.format(ep,np.round(val_loss_sum.item()/b,2),np.round(val_acc_sum.item()/b,2)))
                
     
        
       
#let define some parameters
#batch_size=100
#epoch=10
#one_epoch consist of 50,000/100 iteration
#10 epochs consist of 10* 50,000 /100 iterations
#so total iterations will be 5000
#num_epochs=5000/(50000/100)    
        
epoch=5
iterations=epoch*(len(train)/batch_size)





















