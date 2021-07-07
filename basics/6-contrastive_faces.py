# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 15:59:19 2020

@author: TAC
"""


import os
import torch
import torchvision
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
import tarfile
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader



def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device(type='cpu')

device = get_default_device()
print('default device is',device)


data_dir='D:/tutorials/pytorch/examples/faces/'
print('folder name',os.listdir(data_dir))
print('folder inside train data',os.listdir(data_dir+'training'))
print('folder inside validation data',os.listdir(data_dir+'validation'))
print('folder inside test data',os.listdir(data_dir+'testing'))

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor,Compose,RandomHorizontalFlip,Resize

train_dataset=ImageFolder(root=data_dir+'training')
val_dataset=ImageFolder(root=data_dir+'validation')
test_dataset=ImageFolder(root=data_dir+'testing')
print('Total Images in training data',len(train_dataset))
print('Total Images in validation data',len(val_dataset))
print('Total Images in testing data',len(test_dataset))
      

from helper.contrastive_dataloader import SiameseNetworkDataset
siamese_train =SiameseNetworkDataset(train_dataset,image_D='2D',
                                     transform=Compose([Resize((100,100)),ToTensor()]))
siamese_val =SiameseNetworkDataset(val_dataset,image_D='2D',
                                     transform=Compose([Resize((100,100)),ToTensor()]))

siamese_test =SiameseNetworkDataset(test_dataset,image_D='2D',
                                    transform=Compose([Resize((100,100)),ToTensor()]))


train_loader = DataLoader(siamese_train,shuffle=True,num_workers=0,batch_size=16)

val_loader = DataLoader(siamese_val,shuffle=False,num_workers=0,batch_size=16)

test_loader = DataLoader(siamese_test,shuffle=False,num_workers=0,batch_size=1)

batch = iter(train_loader)
example_batch = next(batch)
# example_batch[0] #image1
# example_batch[1] #image2
# example_batch[2] #label
print(example_batch[0].shape)
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
plt.imshow(torchvision.utils.make_grid(concatenated,nrow=16).permute(1, 2, 0))
#print(example_batch[2].numpy())


from helper.vgg_contrastive import siamese_cnn
from helper.contrastive_loss import ContrastiveLoss
model=siamese_cnn().to(device)
criterion = ContrastiveLoss()
opt=torch.optim.Adam(params=model.parameters(),lr=0.0001)


print('training initiated')
from time import time
train_loss_plt=[]
val_loss_plt=[]
for ep in range(15):
    start=time()
    for train_img1,train_img2,train_labels in train_loader:
        #print(img1.shape,img2.shape,labels.shape)
        #break
        train_img1,train_img2,train_labels = train_img1.to(device),train_img2.to(device),train_labels.to(device)
        train_pred_img1,train_pred_img2=model(train_img1),model(train_img2)
        train_loss=criterion(train_pred_img1,train_pred_img2,train_labels)
        opt.zero_grad()
        train_loss.backward()
        opt.step()
    with torch.no_grad():#essential for valing!!!!    
        val_loss_sum=torch.tensor(0.)
        for val_img1,tet_img2,val_labels in val_loader:
            val_img1,tet_img2,val_labels=val_img1.to(device),tet_img2.to(device),val_labels.to(device)
            val_pred1,val_pred2=model(val_img1),model(tet_img2)
            val_loss=criterion(val_pred1,val_pred2,val_labels)
            val_loss_sum+=val_loss
    final_loss=np.round(val_loss_sum.item()/len(val_loader),2)
    time_taken=np.round((time()-start)/60,2)
    print('Time for Epoch {} is {} minute and train and val loss is {},{} '.format(ep,time_taken,np.round(train_loss.item(),2),final_loss))
    train_loss_plt.append(np.round(train_loss.item(),2))    
    val_loss_plt.append(final_loss)     
    


plt.plot(train_loss_plt,label='training loss')
plt.plot(val_loss_plt,label='validation loss')
plt.legend()
plt.show()

d=torch.nn.PairwiseDistance()
fig,ax=plt.subplots(nrows=5)
for i,test_batch in enumerate (test_loader):
    if i==5:
        break
    img1,img2,_=test_batch
    concatenated = torch.cat((img1,img2),0)
    output1,output2 = model(img1.to(device)),model(img2.to(device))
    dist=d(output1,output2)
    ax[i].imshow(torchvision.utils.make_grid(concatenated).permute(1,2,0))
    ax[i].set_xticks([],[])
    ax[i].set_yticks([],[])
    ax[i].set_title('Dissimilarity: {:.2f}'.format(dist.item()))

















