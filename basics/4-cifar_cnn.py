# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 22:25:58 2020

@author: TAC
"""


import os
import torch
import torchvision
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
import tarfile

# dataset_url="http://files.fast.ai/data/cifar10.tgz"
# download_url(url=dataset_url, root='cifar/')

# with tarfile.open('cifar/cifar10.tgz','r:gz') as tar:
#     tar.extractall(path='cifar')


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device(type='cpu')

device = get_default_device()
print('default device is',device)


data_dir='cifar/cifar10/'
print('folder name',os.listdir(data_dir))
print('folder inside train data',os.listdir(data_dir+'train'))
print('folder inside test data',os.listdir(data_dir+'test'))

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

train_dataset=ImageFolder(root=data_dir+'train',transform=ToTensor())
test_dataset=ImageFolder(root=data_dir+'test',transform=ToTensor())
print('Total Images in training data',len(train_dataset))
print('Total Images in test data',len(test_dataset))


img,label=train_dataset[0]
print(img.shape,label)

print('classes data', train_dataset.classes)
print('classes data',test_dataset.classes)

import matplotlib.pyplot as plt
def show_example(img,label):
    plt.imshow(img.permute(1,2,0))
    plt.title(str(label) + ' : '+train_dataset.classes[label])

show_example(img,label)

tds,vds=random_split(train_dataset, lengths=[int(len(train_dataset)*0.8),
                                             int(len(train_dataset)*0.2)])
print('no of images in train dataset is ',len(tds), 
      'and val dataset is',len(vds))

from torch.utils.data.dataloader import DataLoader
batch_size=128

train_loader=DataLoader(tds,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
val_loader=DataLoader(vds,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)


from torchvision.utils import make_grid

def show_batch(dl):
    batch = next(iter(dl))
    images, labels=batch[0],batch[1]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))

show_batch(train_loader)


def accuracy(pred,actual):
    max_prob,pred=torch.max(pred,dim=1)
    acc=torch.true_divide(torch.sum(pred==actual), len(pred))
    return acc

import torch.nn as nn
class cifar_cnn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=1,padding=1)
        self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv3=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=1,padding=1)
        self.conv4=nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,stride=1,padding=1)
        self.maxpool2=nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.flatten=nn.Flatten()
        self.fcLayer1=nn.Linear(in_features=16384,out_features=100)
        self.fcLayer2=nn.Linear(in_features=100,out_features=10)

        self.relu=nn.ReLU()
        self.softmax=nn.Softmax(dim=1)
        
    def forward(self,x):
        x=self.conv1(x)
        
        x=self.relu(x)
        x=self.conv2(x)
        x=self.relu(x)
        x=self.maxpool1(x)
        
        x=self.conv3(x)
        x=self.relu(x)
        x=self.conv4(x)
        x=self.relu(x)
        x=self.maxpool2(x)
        x=x.view(x.size(0),-1)
        
        x=self.fcLayer1(x)
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
    
    def test_step(self,test_image):
        image=test_image.to(device)
        pred=self(image)
        return pred


model = cifar_cnn().to(device)
model

#this code is to test the architecture
for training_batch in train_loader:
    out=model.training_step(training_batch)
    break

opt=torch.optim.Adam(params=model.parameters(),lr=0.0001)
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

plt.plot(ep_loss)
plt.plot(ep_acc)




