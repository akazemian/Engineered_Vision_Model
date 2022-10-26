import torch
from torch import nn
import numpy as np
from sklearn.decomposition import PCA
from tools.loading import LoadImagePaths
from torchvision import transforms
from random import sample,seed
import os
import logging
import h5py
from PIL import Image
from tools.processing import *
import pickle
import sys
sys.path.append('/home/akazemi3/Desktop/MB_Lab_Project')
from models.model_layers.convolution import *
from models.model_layers.interactions import Interactions
from models.model_layers.output import Output
import math 
    

class _PCA(nn.Module):
    
    def __init__(self,n_components=1000, svd_solver='auto'):
        super().__init__()
    
        self.n_components = n_components
        self.svd_solver = svd_solver

    
    def forward(self,x):
       
        x = torch.Tensor(x).data.cpu().numpy()
        

#         print('pca loop start')
#         img_paths = LoadImagePaths('imagenet21k')
#         images = PreprocessGS(img_paths) 
#         pca = PCA(n_components=self.n_components)
#         model = EngineeredModel(pca=True).Build()
#         output = model(images)
#         print('shape before pca',output.shape)
#         pca.fit(output)
#         pickle.dump(pca, open("pca.pkl","wb"))
        
        
        pca = pickle.load(open("/home/akazemi3/Desktop/MB_Lab_Project/models/pca.pkl",'rb'))

        return torch.Tensor(pca.transform(x))
