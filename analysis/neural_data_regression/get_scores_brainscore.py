import sys
# path = '/home/akazemi3/.conda/envs/bonner-lab-atlas/lib/python3.7/site-packages/'
# sys.path.append(path)
# from kymatio.torch import Scattering2D
import os 
path = '/home/akazemi3/Desktop/MB_Lab_Project/'
sys.path.append(path)
results_path = os.path.join(path, 'Results/Encoding_Performance')
import torch 
from torch import nn
import numpy as np
from sklearn import linear_model
from scipy.stats import pearsonr
from model_tools.activations import PytorchWrapper
import math
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import brainscore.benchmarks as bench
from brainscore.metrics.regression import linear_regression, ridge_regression
from model_tools.brain_transformation.neural import LayerScores
import matplotlib.pyplot as plt
from pytorch_hmax import hmax
model_name = 'hmax'
from sklearn.linear_model import Ridge
from tools.processing import *
#from models.call_model import *
from tools.loading import *
# from scattering_transform.scattering.ST import *
import importlib
import random
import shutil
alpha = 1
model_scores_path = '/data/atlas/model_scores'
os.environ['BRAINIO_HOME']= "/data/shared/.cache/brainscore/brainio"




data = {'name':'dicarlo.MajajHong2015public.','regions':['V4','IT']}
#data = {'name':'movshon.FreemanZiemba2013public.','regions':['V1','V2']}

# model = Scattering2D(J=5, shape=(96, 96))
# model_name = 'scattering_J=5'
# layers = ['logits']  

# model_name = 'alexnet'
# model = torchvision.models.alexnet(pretrained=True)
# layers = ['features.12']

# model = EngineeredModel2Gabor().Build()
# model_name = 'engineered_eng2_gabor_'
# layers = ['logits'] 


# model = EngineeredModel2().Build()
# model_name = 'engineered_sc_2'
# layers = ['logits']  

# model = hmax.HMAX(universal_patch_set=os.path.join(path,'pytorch_hmax/universal_patch_set.mat'))
# model_name = 'hmax'
# layers = ['logits'] 


# model_name = 'scattering_sc'
# H, W = 96, 96
# model = ScatteringTransform(M=H, N=W, J=4, L=8).Build()
# layers = ['s2'] 


# model = kymatio2d(J=2, input_shape=(96, 96),layer='all').Build()
# model_name = 'kymatio_J=2'
# layers = ['logits']  


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
from models.all_models.model_4convlayer_onlyconv_manycurvfilt_nonl import EngineeredModel4COnlyCManyCurvFiltNoNL
from models.all_models.random_features_normal import RandomFeaturesN
from models.all_models.random_features_uniform import RandomFeaturesU
from models.all_models.model_4convlayer_onlyconv_manycurvfilt_nolargefilt import EngineeredModel4COnlyCManyCurvFiltNoLFilt
from models.all_models.model_1convlayer_onlyconv_manycurvfilt_largefilt import EngineeredModel1COnlyCManyCurvFiltLFilt
from models.all_models.model_1convlayer_onlyconv_manycurvfilt_smallfilt import EngineeredModel1COnlyCManyCurvFiltSFilt
from models.all_models.model_4convlayer_onlyconv_manycurvfilt_nonl_avgpool import EngineeredModel4COnlyCManyCurvFiltNoNLAvgPool

# model_list = {'model_4convlayer_alllayers':EngineeredModel4CAll,
#               'model_4convlayer_ints2layer':EngineeredModel4CInts2, 
#               'model_1convlayer_alllayers':EngineeredModel1CAll, 
#               'model_1convlayer_ints2layers':EngineeredModel1CInts2,
#               'alexnet_alllayers':AlexNet,
#               'model_1convlayer_ints2layers_largelayer1filts':EngineeredModel1CInts2MoreFilters,
#               'model_4convlayer_ints2layer_randomlayer1filts':EngineeredModel4CInts2RandomL1,
#               'model_4convlayer_conv2layer':EngineeredModel4CConv2,
#               'model_4convlayer_onlyconv_curvfilt':EngineeredModel4COnlyCCurvFilt,
#               'model_4convlayer_onlyconv_manycurvfilt':EngineeredModel4COnlyCManyCurvFilt,
#               'model_4convlayer_onlyconv_manycurvfilt_nonl':EngineeredModel4COnlyCManyCurvFiltNoDN,
#               'random_features_normal':RandomFeaturesN,
#                'random_features_uniform_1':RandomFeaturesU

#              }

model_list = {
            'model_4convlayer_onlyconv_manycurvfilt_nonl_2':EngineeredModel4COnlyCManyCurvFiltNoNL().Build()
    
             }

for model_name, model in model_list.items():
    
    layers=['last']


    model_iden = model_name + '_' + data['name']
    preprocess = PreprocessRGB if 'alexnet' in model_name else PreprocessGS



    if os.path.exists(os.path.join(model_scores_path,f'{model_iden}_Ridge(alpha={alpha})')):
        print(f'model scores are already saved in {model_scores_path} as {model_iden}_Ridge(alpha={alpha})')

    else:
        print('obtaining model scores...')
        ds = xr.Dataset(data_vars=dict(r_value=(["r_values"], [])),
                                    coords={'region': (['r_values'], [])
                                                 })
        for region in data['regions']: 

            benchmark_iden = data['name'] + region + '-pls'

            activations_model = PytorchWrapper(model=model,
                                               preprocessing=preprocess,
                                               identifier=model_iden)

            benchmark = bench.load(benchmark_iden)
            benchmark._identifier = benchmark.identifier.replace('pls', f'ridge_alpha={alpha}')
            benchmark._similarity_metric.regression = ridge_regression(sklearn_kwargs = {'alpha':alpha})


            model_scores = LayerScores(model_identifier=activations_model.identifier,
                                   activations_model=activations_model,
                                   visual_degrees=8)

            score = model_scores(benchmark=benchmark,layers=layers,prerun=True)
            r_values = score.raw.raw.values.reshape(-1)
            ds_tmp = xr.Dataset(data_vars=dict(r_value=(["r_values"], r_values)),
                                                coords={'region': (['r_values'], 
                                                                   [region for i in range(len(r_values))])
                                                             })

            ds = xr.concat([ds,ds_tmp],dim='r_values')


        ds.to_netcdf(os.path.join(model_scores_path,f'{model_iden}_Ridge(alpha={alpha})'))
        print(f'model scores are now saved in {model_scores_path} as {model_iden}_Ridge(alpha={alpha})')