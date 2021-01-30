# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 14:10:14 2021
https://jovian.ai/learn/deep-learning-with-pytorch-zero-to-gans/lesson/lesson-6-image-generation-using-gans
@author: TAC
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]
def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break



from time import time
class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(    
            nn.ConvTranspose2d(z_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)
    
    

   
# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 0.0002
z_dim = 128
image_dim = 64
batch_size = 256
num_epochs = 25  

gen = Generator(z_dim, image_dim) 

# xb = torch.randn(batch_size, z_dim, 1, 1) # random latent tensors
# fake_images = gen(xb)
# print(fake_images.shape)
# show_images(fake_images)




disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device) 


fixed_noise = torch.randn((batch_size, z_dim)).to(device)
aug = transforms.Compose(
    [transforms.Resize((image_dim,image_dim)),transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),]
)

dataset = datasets.ImageFolder(root="dataset/anime_image/", transform=aug)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr,betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=lr,betas=(0.5, 0.999))
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
writer_time=SummaryWriter(f"logs/time")
step = 0

fixed_noise = torch.randn((batch_size, z_dim,1,1)).to(device)

def train_discriminator(real_images, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = disc(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss =  criterion(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()
    
    # Generate fake images
    latent = torch.randn(batch_size, z_dim, 1, 1, device=device)
    fake_images = gen(latent)

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = disc(fake_images)
    fake_loss = criterion(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score


def train_generator(opt_g):
    # Clear generator gradients
    opt_g.zero_grad()
    
    # Generate fake images
    latent = torch.randn(batch_size, z_dim, 1, 1, device=device)
    fake_images = gen(latent)
    
    # Try to fool the discriminator
    preds = disc(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = criterion(preds, targets)
    
    # Update generator weights
    loss.backward()
    opt_g.step()
    
    return loss.item()

for epoch in range(num_epochs):
    start=time()
    for batch_idx, (real, _) in enumerate(loader):

        lossD, real_score, fake_score = train_discriminator(real.to(device) , opt_disc)
        lossG = train_generator(opt_gen)
    end=time()
    epoch_time=end-start
    print(
                f"Epoch {epoch} Time: {epoch_time:.4f} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )
    writer_time.add_scalar('Time logs', epoch_time,global_step=step)
    writer_time.add_scalar('Disc loss', lossD,global_step=step)
    writer_time.add_scalar('Gen loss', lossG,global_step=step)
    #writer_param.add_hparams({hparam_dict}, metric_dict)
    with torch.no_grad():
        fake = gen(fixed_noise).reshape(-1,3, 64, 64)
        data = real.reshape(-1, 3, 64, 64)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
        img_grid_real = torchvision.utils.make_grid(data, normalize=True)

        writer_fake.add_image(
            "Fake Images", img_grid_fake, global_step=step
        )
        writer_real.add_image(
            "Real Images", img_grid_real, global_step=step
        )
        step += 1