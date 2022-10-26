# getting activations for a specific dataset from a specific model. Output is an xarray with dims: features x presentation (stimulus_id)
# from kymatio.torch import Scattering2D
import xarray as xr
import os 
import sys
import torchvision

path = '/home/akazemi3/Desktop/MB_Lab_Project/'
sys.path.append(path)
# from models.kymatio import *

from tools.processing import *
from models.call_model import *
from tools.loading import *
from analysis.regression.tools.extractor import Activations
from pytorch_hmax import hmax
from scipy.io import loadmat
import xarray as xr
import numpy as np
import torch
import matplotlib.pyplot as plt
from analysis.regression.tools.regression import *
from analysis.regression.tools.metrics import *
results_path = '/data/atlas/activations'




dataset_name = 'naturalscenes'
#regions = ['roi_EVC','roi_LOC']
regions = ['roi_prf-visualrois_V1v','roi_prf-visualrois_V2v','roi_prf-visualrois_V3v','roi_prf-visualrois_hV4']


# model = Scattering2D(J=5, shape=(96, 96))
# model_name = 'scattering_J=5'
# layers = ['logits']  
# model = kymatio2d(J=2, input_shape=(96, 96),layer='all').Build()
# model_name = 'kymatio_J=2'
# layers = ['logits']  

# model = hmax.HMAX(universal_patch_set=os.path.join(path,'pytorch_hmax/universal_patch_set.mat'))
# model_name = 'hmax'
# layers= ['logits'] 


# model = EngineeredModel().Build()
# model_name = 'engineered1'
# layers = ['logits'] 

# from models.all_models.alexnet_custom import AlexNet
# model = AlexNet().Build()
# model_name = 'alexnet_alllayers'
# layers = ['last'] 


# model = kymatio2d(J=4, input_shape=(96, 96),layer='all').Build()
# model_name = 'kymatio_all_nl'
# layers = ['logits']  

# 'engineered_st_3' has 3 diff scales and 'engineered_st_4' has 4 diff scales

# model = torchvision.models.alexnet(pretrained=True)
# model_name = 'alexnet'
# layers = ['features.12']  


# H, W = 96, 96
# model = ScatteringTransform(M=H, N=W, J=4, L=8).Build()
# model_name = 'scattering_sc'
# layers = ['s2'] 

from models.all_models.model_4convlayer_alllayers import EngineeredModel4CAll
from models.all_models.model_4convlayer_ints2layer import EngineeredModel4CInts2
from models.all_models.model_1convlayer_alllayers import EngineeredModel1CAll
from models.all_models.model_1convlayer_ints2layer import EngineeredModel1CInts2 
from models.all_models.alexnet_custom import AlexNet
from models.all_models.model_1convlayer_ints2layer_largelayer1filts import EngineeredModel1CInts2MoreFilters 
from models.all_models.model_4convlayer_ints2layer_randomlayer1filts import EngineeredModel4CInts2RandomL1
from models.all_models.model_4convlayer_conv2layer import EngineeredModel4CConv2
from models.all_models.model_4convlayer_onlyconv_curvfilt import EngineeredModel4COnlyCCurvFilt
from models.all_models.model_4convlayer_onlyconv_manycurvfilt import EngineeredModel4COnlyCManyCurvFilt
from models.all_models.model_4convlayer_onlyconv_3layer import EngineeredModel4COnlyCCurvFilt3L
from models.all_models.model_4convlayer_onlyconv_manycurvfilt_nodn import EngineeredModel4COnlyCManyCurvFiltNoDN

model_list = {
              'alexnet_alllayers':AlexNet,
              'model_4convlayer_onlyconv_curvfilt':EngineeredModel4COnlyCCurvFilt,
              'model_4convlayer_onlyconv_manycurvfilt':EngineeredModel4COnlyCManyCurvFilt,
    'model_4convlayer_onlyconv_manycurvfilt_nodn':EngineeredModel4COnlyCManyCurvFiltNoDN,
    'model_4convlayer_onlyconv_manycurvfilt_nonl':EngineeredModel4COnlyCManyCurvFiltNoDN,
             }



for model_name, func in model_list.items():
    
    model = func().Build()
    layers=['last']


    preprocess = PreprocessRGB if 'alexnet' in model_name else PreprocessGS

    images = LoadImagePaths(dataset_name)
    processed_images = preprocess(images) #change based on model
    image_labels = [os.path.basename(i).strip('.png') for i in images]


    iden = model_name + '_' + dataset_name                  
    # get activations  
    activations = Activations(model=model,
                        layer_names=layers,
                        images=processed_images,
                        image_labels=image_labels,
                        )                   
    activations.get_array(results_path,iden)     


    # get model scores
    r_values = get_rvalues(identifier=iden,
                           dataset=dataset_name,
                           regions=regions,
                           ridge_alpha=1)