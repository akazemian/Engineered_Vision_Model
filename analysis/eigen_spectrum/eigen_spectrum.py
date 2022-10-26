import sys
path = '/home/akazemi3/.conda/envs/bonner-lab-atlas/lib/python3.7/site-packages/'
sys.path.append(path)
from kymatio.torch import Scattering2D
path = '/home/akazemi3/Desktop/MB_Lab_Project/'
sys.path.append(path)
from analysis.eigen_spectrum.analyses import *
from model_tools.activations import PytorchWrapper
from tools.loading import *
from tools.processing import *
from models.call_model import *

from analysis.eigen_spectrum.hooks import LayerGlobalMaxPool2d
from unittest.mock import patch

# param_list= ['NO_NL',
#             'NO_ABS',
#             'NO_ZSCORE',
#             'RELU_AND_ZSCORE',
#             'NO_LAYER_1_INTS',
#             'NO_LAYER_2_INTS',
#             'NO_LAYER_INTS']

#param = param_list[1]


model = Scattering2D(J=4, shape=(96, 96))
layers = ['logits'] 


model_name = 'scattering'
param = 'J=4'
iden= model_name + '_' + param




with patch.dict("os.environ", {"RESULTCACHING_HOME": '/home/akazemi3/Documents/cache_temp'}):
    
    os.environ['RESULTCACHING_DISABLE'] = '1'

    results_path = os.path.join(path,'results', 'eigen_spectrum', f'{model_name}_{param}.nc')
    

    activations_model = PytorchWrapper(model=model, preprocessing=PreprocessGS,identifier=iden)

    _ = LayerGlobalMaxPool2d.hook(activations_model, identifier='')

    activations_extractor = activations_model  # Some PyTorchWrapper
    stimuli = LoadImagenet21kVal()        # Either a list of stimuli paths or a StimulusSet data type
    
    batch_size = None            # None if you have a lot of RAM and don't want to use batched PCA
    stimuli_identifier = iden    # String used for caching the results. Set for False to disable caching

    eigspec_analysis = ModelEigspecAnalysis(activations_extractor)
    results = eigspec_analysis(stimuli=stimuli,
                            layers=layers,
                            stimuli_identifier=stimuli_identifier,
                            batch_size=batch_size)

    #results.to_netcdf(results_path)
    results.to_netcdf(f'/data/atlas/{model_name}_{param}.nc')
