
import warnings
warnings.warn('my warning')

from collections import OrderedDict
import xarray as xr

import logging
import numpy as np
from PIL import Image

SUBMODULE_SEPARATOR = '.'
from torch import nn


SUBMODULE_SEPARATOR = '.'
import os
import torch
from torch.autograd import Variable


class PytorchWrapper:
    def __init__(self, model,forward_kwargs=None): #preprocessing=None, identifier=None,  *args, **kwargs

        #logger = logging.getLogger(fullname(self))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #logger.debug(f"Using device {self._device}")
        self._model = model
        self._model = self._model.to(self._device)
        self._forward_kwargs = forward_kwargs or {}


    def get_activations(self, images, layer_names):

        images = [torch.from_numpy(image) if not isinstance(image, torch.Tensor) else image for image in images]
        images = Variable(torch.stack(images))
        images = images.to(self._device)
        self._model.eval()

        layer_results = OrderedDict()
        hooks = []

        for layer_name in layer_names:
            layer = self.get_layer(layer_name)
            hook = self.register_hook(layer, layer_name, target_dict=layer_results)
            hooks.append(hook)

        with torch.no_grad():
            self._model(images, **self._forward_kwargs)
        for hook in hooks:
            hook.remove()
        return layer_results

    def get_layer(self, layer_name):
        if layer_name == 'logits':
            return self._output_layer()
        module = self._model
        for part in layer_name.split(SUBMODULE_SEPARATOR):
            module = module._modules.get(part)
            assert module is not None, f"No submodule found for layer {layer_name}, at part {part}"
        return module

    def _output_layer(self):
        module = self._model
        while module._modules:
            module = module._modules[next(reversed(module._modules))]
        return module

    @classmethod
    def _tensor_to_numpy(cls, output):
        try:
            return output.cpu().data.numpy()
        except AttributeError:
            return output
            

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            target_dict[name] = PytorchWrapper._tensor_to_numpy(output)

        hook = layer.register_forward_hook(hook_function)
        return hook

    def __repr__(self):
        return repr(self._model)
    
    

    
def batch_activations(model, layer_names, images, image_labels):

        
        activations_dict = model.get_activations(images=images, layer_names=layer_names)

        da_list = []
        for layer in layer_names:
            activations = activations_dict[layer].reshape(activations_dict[layer].shape[0],-1)
            ds = xr.Dataset(
                data_vars=dict(x=(["presentation", "features"], activations)),
                coords={'stimulus_id': (['presentation'], image_labels)})
            da_list.append(ds)

        data = xr.concat(da_list,dim='presentation') 
        return data
    
    
class Activations:
    
    def __init__(self,model,layer_names,images,image_labels):
        
        self.model = model
        self.layer_names = layer_names
        self.images = images
        self.image_labels = image_labels
     
        
    def get_array(self,path,identifier):
        if os.path.exists(os.path.join(path,identifier)):
            print(f'array is already saved in {path} as {identifier}')
        
        else:
        
            print('extracting activations')
            wrapped_model = PytorchWrapper(self.model)

            i = 0
            batch = 100
            ds_list = []           
            
            while i <= len(self.images):
                
                batch_data = batch_activations(wrapped_model, self.layer_names, self.images[i:i+batch], self.image_labels[i:i+batch])
                ds_list.append(batch_data)
                i += batch
            
            
            data = xr.concat(ds_list,dim='presentation')
            
            if identifier.split('_')[-1] == 'object2vec':
                data = data.assign_coords(stimulus_id=(data.stimulus_id.str.slice_replace(-3)))
                data = data.groupby('stimulus_id').mean()
                
            data.to_netcdf(os.path.join(path,identifier))
            print(f'array is now saved in {path} as {identifier}')