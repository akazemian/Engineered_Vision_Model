# import sys
# path = '/home/akazemi3/.conda/envs/bonner-lab-atlas/lib/python3.7/site-packages/'
# sys.path.append(path)
# from kymatio.torch import Scattering2D
import sys
sys.path.append('/home/akazemi3/Desktop/MB_Lab_Project')
from models.engineered_models import *
from models.model_layers.convolution import *
from models.model_layers.nonlinearity import nonlinearity
from models.model_layers.interactions import Interactions
from models.model_layers.output import Output
from models.model_layers.pca import _PCA
import math 



class EngineeredModel:
    
    def __init__(self,curv_params = {'n_ories':3,'n_curves':8,'gau_sizes':(5,),'spatial_fre':[1.2]},
                 filters_2=20000):
    
        self.curv_params = curv_params
        self.filters_1 = self.curv_params['n_ories']*self.curv_params['n_curves']*len(self.curv_params['gau_sizes']*len(self.curv_params['spatial_fre']))
        self.filters_2 = filters_2
    
    
    
    
    def Build(self):
    
        c1_1 = StandardConvolution(filter_size=72,filter_type='Curvature',pooling=('max',6),curv_params=self.curv_params)
        c1_2 = StandardConvolution(filter_size=45,filter_type='Curvature',pooling=('max',6),curv_params=self.curv_params)
        c1_3 = StandardConvolution(filter_size=33,filter_type='Curvature',pooling=('max',6),curv_params=self.curv_params)
        c1_4 = StandardConvolution(filter_size=21,filter_type='Curvature',pooling=('max',6),curv_params=self.curv_params)
        
                
        
        c2_1 = StandardConvolution(out_channels=self.filters_2,filter_size=15,filter_type='Random',pooling=('max',8))
        c2_2 = StandardConvolution(out_channels=self.filters_2,filter_size=9,filter_type='Random',pooling=('max',8))
        c2_3 = StandardConvolution(out_channels=self.filters_2,filter_size=5,filter_type='Random',pooling=('max',8))
        c2_4 = StandardConvolution(out_channels=self.filters_2,filter_size=3,filter_type='Random',pooling=('max',8))
        
        last = Output()

        return InteractionsModel(c1_1,c1_2,c1_3,c1_4,c2_1,c2_2,c2_3,c2_4,last)  
    
    
    
    
    

    
    
    

