import sys
sys.path.append('/home/akazemi3/Desktop/MB_Lab_Project')
from models.model_layers.convolution import *
from models.model_layers.nonlinearity import nonlinearity
from models.model_layers.interactions import Interactions
from models.model_layers.output import Output
from models.model_layers.pca import _PCA
import math 
from multiprocessing.sharedctypes import Value
import torch
from torch import nn
from torchvision.transforms.functional import gaussian_blur
import numpy as np
                         

class InteractionsModel(nn.Module):
    
    def __init__(self,
                c1_1: nn.Module,
                c1_2: nn.Module,
                c1_3: nn.Module,
                c1_4: nn.Module,
                c2_1: nn.Module,
                c2_2: nn.Module,
                c2_3: nn.Module,
                c2_4: nn.Module,
                last : nn.Module
                ):
        
        super(InteractionsModel, self).__init__()
        

        self.c1_1 = c1_1
        self.c1_2 = c1_2
        self.c1_3 = c1_3
        self.c1_4 = c1_4
                
        
        self.c2_1 = c2_1
        self.c2_2 = c2_2
        self.c2_3 = c2_3
        self.c2_4 = c2_4
        
        self.mp = nn.MaxPool2d(2)
        self.last = last
        
        
    def forward(self, x_orig:nn.Module):
        

        
        N = x_orig.shape[0]
        #conv layer 1
        x_c1_1 = self.c1_1(x_orig) # c1_1 smallest filter
        x_c1_2 = self.c1_2(x_orig) # c1_2 
        x_c1_3 = self.c1_3(x_orig) # c1_3 
        x_c1_4 = self.c1_4(x_orig) # c1_3 
        x_c1 = torch.cat([x_c1_1,x_c1_2,x_c1_3,x_c1_4],dim=1)
        print('conv1', x_c1.shape)
        
        #x_c1 = self.mp(x_c1)    
        
        #conv layer 2
        x_c2_1 = self.c2_1(x_c1) 
        x_c2_2 = self.c2_2(x_c1) 
        x_c2_3 = self.c2_3(x_c1) 
        x_c2_4 = self.c2_4(x_c1)
        x_c2 = torch.cat([x_c2_1,x_c2_2,x_c2_3,x_c2_4],dim=1)      
        print('conv2', x_c2.shape) 
        
        #x_c2 = self.mp(x_c2)
        
        x = self.last(x_c2)
        
        print('output', x.shape)
        return x    



  