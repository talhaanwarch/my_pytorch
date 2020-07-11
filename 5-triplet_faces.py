# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 11:55:26 2020

@author: TAC
"""


import os
import torch
import torchvision
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
import tarfile
import numpy as np

import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor,Compose,RandomHorizontalFlip,Resize
from torch.utils.data.dataloader import DataLoader

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device(type='cpu')

device = get_default_device()
print('default device is',device)


data_dir='D:/tutorials/pytorch/examples/faces/'
train_dataset=ImageFolder(root=data_dir+'training')
val_dataset=ImageFolder(root=data_dir+'validation')
test_dataset=ImageFolder(root=data_dir+'testing')


from helper.triplet_dataloader import SiameseNetworkDataset

siamese_train =SiameseNetworkDataset(train_dataset,image_D='2D',
                                     transform=Compose([Resize((100,100)),ToTensor()]))

train_loader = DataLoader(siamese_train,shuffle=True,num_workers=0,batch_size=16)



siamese_val =SiameseNetworkDataset(val_dataset,image_D='2D',
                                     transform=Compose([Resize((100,100)),ToTensor()]))

val_loader = DataLoader(siamese_val,shuffle=False,num_workers=0,batch_size=16)



#for testing and plot puprose whether out SiameseNetworkDataset class is working or not
batch = iter(train_loader)
example_batch = next(batch)
concatenated = torch.cat((example_batch[0],example_batch[1],example_batch[2]),0)
plt.imshow(torchvision.utils.make_grid(concatenated,nrow=16).permute(1, 2, 0))

from helper.vgg_triplet import siamese_vgg16
model=siamese_vgg16().to(device)


from helper.loss_triplet import Triplet_loss

criterion = Triplet_loss()
opt=torch.optim.Adam(params=model.parameters(),lr=0.0001)


from time import time
train_loss_plt=[]
val_loss_plt=[]
for ep in range(15):
    start=time()
    for train_batch in train_loader:
        anchor, positive, negative=train_batch
        anchor, positive, negative= anchor.to(device), positive.to(device), negative.to(device)
        anchor_out,positive_out,negative_out = model(anchor,positive,negative)
        #print(anchor_out.shape, positive_out.shape, negative_out.shape)
        train_loss=criterion(anchor_out,positive_out,negative_out)
        opt.zero_grad()
        train_loss.backward()
        opt.step()
        with torch.no_grad():#essential for valing!!!!    
            val_loss_sum=torch.tensor(0.)
            
            for val_anchor,val_positive,val_negative in val_loader:
                val_anchor,val_positive,val_negative=val_anchor.to(device),val_positive.to(device),val_negative.to(device)
                val_pred1,val_pred2,val_pred3=model(val_anchor,val_positive,val_negative)
                val_loss=criterion(val_pred1,val_pred2,val_pred3)
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
    

torch.save(model.state_dict(), 'D:/tutorials/pytorch/examples/triplet_checkpoint.pth')

model.load_state_dict(torch.load('D:/tutorials/pytorch/examples/triplet_checkpoint.pth'))



siamese_test =SiameseNetworkDataset(test_dataset,image_D='2D',
                                     transform=Compose([Resize((100,100)),ToTensor()]))

test_loader = DataLoader(siamese_test,shuffle=False,num_workers=0,batch_size=1)
d = torch.nn.PairwiseDistance(p=2)
fig,ax=plt.subplots(nrows=5)
for i,test_batch in enumerate (test_loader):
    if i==5:
        break
    test_anchor,test_pos,test_neg=test_batch
    concatenated = torch.cat((test_anchor,test_pos,test_neg),0)
    test_anchor,test_pos,test_neg=test_anchor.to(device),test_pos.to(device),test_neg.to(device)
    test_pred1,test_pred2,test_pred3=model(test_anchor,test_pos,test_neg)
    apd=np.round(d(test_pred1,test_pred2).item(),3) #distanc between anchor and positive
    anp=np.round(d(test_pred1,test_pred3).item(),3) #distance between anchor and negative

    ax[i].imshow(torchvision.utils.make_grid(concatenated,nrow=3).permute(1,2,0))
    ax[i].set_title('AP dist {} & AN dist {}'.format(apd,anp))
    ax[i].set_xticks([],[])
    ax[i].set_yticks([],[])








