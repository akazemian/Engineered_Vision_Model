# getting activations for a specific dataset from a specific model. Output is an xarray with dims: features x presentation (stimulus_id)


import xarray as xr
import os 
import sys
import torchvision
path = '/home/akazemi3/Desktop/MB_Lab_Project/'
sys.path.append(path)
from tools.Processing import *
from models.Call_Model import *
from tools.Loading import *
from extractor import Activations
from pytorch_hmax import hmax
from scipy.io import loadmat
results_path = '/data/atlas/Activations'




model_name = 'hmax'
dataset_name = 'naturalscenes'
layer_names = ['logits']  
model = hmax.HMAX(universal_patch_set=os.path.join(path,'pytorch_hmax/universal_patch_set.mat'))


# model_name = 'engineered'
# dataset_name = 'object2vec'
# layer_names = ['output']  
# model = Model().Build()

# model_name = 'alexnet'
# dataset_name = 'object2vec'
# layer_names = ['features.12']  # if this runs, then we can just add hmax here and pass logits as the layer name to get the output
# model = torchvision.models.alexnet(pretrained=True)


iden = model_name + '_' + dataset_name


images = LoadImagePaths(dataset_name)
processed_images = PreprocessGS(images) #change based on model
image_labels = [os.path.basename(i).strip('.png') for i in images]
                      

activations = Activations(model=model,
                    layer_names=layer_names,
                    images=processed_images,
                    image_labels=image_labels,
                    )
                        
activations.get_array(results_path,iden)      
