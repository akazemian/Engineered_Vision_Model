from models.layer_operations.convolution import *
from models.layer_operations.output import Output
import torch
from torch import nn
                         


class Model(nn.Module):
    
    
    def __init__(self,
                c1: nn.Module,
                c2: nn.Module,
                batches_2: int,
                last: nn.Module
                ):
        
        super(Model, self).__init__()
        
        
        self.c1 = c1 
        self.c2 = c2
        self.batches_2 = batches_2
        self.last = last
        self.mp = nn.MaxPool2d(2)
        
        
    def forward(self, x:nn.Module):
                
        #conv layer 1
        x = self.c1(x)
        print('conv1', x.shape)
    
        
        #conv layer 2
        conv_2 = []
        for i in range(self.c2_batch):
            conv_2.append(self.c2(x)) 
        x = torch.cat(conv_2,dim=1)
        print('conv2', x.shape)

        
        x = self.last(x)
        print('output', x.shape)
        
        return x    



  