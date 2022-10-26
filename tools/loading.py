from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from random import sample,seed
import os
#import logging
import numpy as np
import h5py
from PIL import Image
import xarray as xr




def LoadObject2VecImages():
    data_path = '/data/shared/datasets/object2vec/stimuli'
    all_images = []
    for folder in os.listdir(data_path):
        for image in os.listdir(f'{data_path}/{folder}'):
            all_images.append(f'{data_path}/{folder}/{image}')
    return all_images    



def LoadImagenet21kImages():
    all_images = []
    num_classes=50
    num_per_class=30
    path = '/data/shared/datasets/imagenet21k_sorscher2021'
    seed(27)
    folders = os.listdir(path)
    cats = sample(folders,num_classes)
    for cat in cats:
        images = os.listdir(os.path.join(path,cat))
        examples = sample(images,num_per_class)
        for example in examples:
            example_path = os.path.join(os.path.join(path,cat,example))
            all_images.append(example_path)
    return all_images
    
    
    
    
def LoadNSDImages(shared_images=True):
    
    path = '/data/shared/datasets/allen2021.natural_scenes/images'
    all_images = []
    if shared_images:
        shared_ids = list(xr.open_dataset('/data/atlas/activations/alexnet_naturalscenes').stimulus_id.values)
        
        for image in sorted(os.listdir(path)):
            if image.strip('.png') in shared_ids:
                all_images.append(f'{path}/{image}')
            
    else:
        for image in sorted(os.listdir(path)):
            all_images.append(f'{path}/{image}')
    #seed(27)
    #images = sample(all_images)
    return all_images 
    

    
    
def LoadImagenet21kVal(num_classes=1000, num_per_class=10, separate_classes=False):
    #_logger = logging.getLogger(fullname(LoadImagenet21kVal))
    base_indices = np.arange(num_per_class).astype(int)
    indices = []
    for i in range(num_classes):
        indices.extend(50 * i + base_indices)

    framework_home = os.path.expanduser(os.getenv('MT_HOME', '~/.model-tools'))
    imagenet_filepath = os.getenv('MT_IMAGENET_PATH', os.path.join(framework_home, 'imagenet2012.hdf5'))
    imagenet_dir = f"{imagenet_filepath}-files"
    os.makedirs(imagenet_dir, exist_ok=True)

    if not os.path.isfile(imagenet_filepath):
        os.makedirs(os.path.dirname(imagenet_filepath), exist_ok=True)
        #_logger.debug(f"Downloading ImageNet validation to {imagenet_filepath}")
        s3.download_file("imagenet2012-val.hdf5", imagenet_filepath)

    filepaths = []
    with h5py.File(imagenet_filepath, 'r') as f:
        for index in indices:
            imagepath = os.path.join(imagenet_dir, f"{index}.png")
            if not os.path.isfile(imagepath):
                image = np.array(f['val/images'][index])
                Image.fromarray(image).save(imagepath)
            filepaths.append(imagepath)

    if separate_classes:
        filepaths = [filepaths[i * num_per_class:(i + 1) * num_per_class]
                    for i in range(num_classes)]
    return filepaths



    
    
def LoadImagePaths(name):
    
    if name == 'object2vec':
        return LoadObject2VecImages()
        
    elif name == 'imagenet21k':
        return LoadImagenet21kImages()

    elif name == 'imagenet21k_val':
        num_classes=1000
        num_per_class=10
        return LoadImagenet21kVal(num_classes=num_classes, num_per_class=num_per_class, separate_classes=False)
    
    elif name =='naturalscenes':
        return LoadNSDImages()
    
