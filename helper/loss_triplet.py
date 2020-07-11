# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 15:37:01 2020

@author: TAC
"""

import torch
#triplet_naive approach
# def triplet_naive(a,p,n,alpha):
#     pos_dist=(a-p).pow(2).sum(1)
#     neg_dist=(a-n).pow(2).sum(1)
#     loss=torch.max(pos_dist-neg_dist+alpha,0).sum(0)
#     return loss
    
class Triplet_loss(torch.nn.Module):
    def __init__(self, alpha=2.0):
        super(Triplet_loss, self).__init__()
        self.alpha = alpha
#https://discuss.pytorch.org/t/triplet-loss-in-pytorch/30634/2
    def forward(self, a, p, n):
        d = torch.nn.PairwiseDistance(p=2)
        distance = d(a, p) - d(a, n) + self.alpha
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance))) 
        return loss