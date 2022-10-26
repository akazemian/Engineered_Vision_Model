from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from random import sample,seed
import os
import logging
import numpy as np
import h5py
from PIL import Image




    
def PreprocessGS(images):
    size = 96
    transform = transforms.Compose([
         transforms.Resize((size,size)),
         transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize(mean=0.5, std=0.5)])
    
    try:
        return np.stack([transform(Image.open(i).convert('RGB')) for i in images])
    except TypeError: 
        return torch.stack([transform(Image.open(i).convert('RGB')) for i in images])
        
    
    
def PreprocessRGB(images):
    
    size = 224
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
        
    transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])
    
    try:
        return np.stack([transform(Image.open(i).convert('RGB')) for i in images])
    except TypeError:
        return torch.stack([transform(Image.open(i).convert('RGB')) for i in images])

        
    
    
