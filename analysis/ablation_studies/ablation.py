import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import shutil
import pandas as pd
from csv import DictWriter

path = '/home/akazemi3/Desktop/MB_Lab_Project/'
sys.path.append(path)

from tools.processing import *
from models.call_model import *

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from brainscore.metrics.regression import linear_regression, ridge_regression
import brainscore.benchmarks as bench
from model_tools.brain_transformation.neural import LayerScores
from model_tools.activations import PytorchWrapper
from bonnerlab_brainscore.benchmarks.object2vec import Object2VecEncoderBenchmark

os.environ['RESULTCACHING_DISABLE'] = '1'


layers = ['output']
local_data_dict = {'Object2Vec':['EVC','LOC']}
brainscore_data_dict = {'movshon.FreemanZiemba2013':['V1','V2'], 'dicarlo.MajajHong2015':['V4','IT']}

target_param = ['NO_NL',
                'NO_ABS',
                'NO_ZSCORE',
                'RELU_AND_ZSCORE',
                'NO_LAYER_1_INTS2',
                'NO_LAYER_2_INTS',
                'NO_LAYER_INTS']

param = target_param[1]

headers = ['region_dataset',param,'mean_score']           
model = Model().Build()
results_path = os.path.join(path,'Results','Parameter_Study',f'{param}.csv')

# Brainscore Data 
for dataset, regions in brainscore_data_dict.items():
    for region in regions:

        activations_model = PytorchWrapper(model=model, 
                                    preprocessing=PreprocessGS,
                                    identifier=f'Model-{dataset}-{param}')

        benchmark = bench.load(f'{dataset}public.{region}-pls')
        benchmark._identifier = benchmark.identifier.replace('pls', 'lin')
        benchmark._similarity_metric.regression = linear_regression()
        benchmark._similarity_metric.regression._regression.alpha = 0.1

        model_scores = LayerScores(model_identifier=activations_model.identifier,
                               activations_model=activations_model,
                               visual_degrees=8)

        try:
            score = model_scores(benchmark=benchmark,layers=layers,prerun=True)

        except FileExistsError:
            shutil.rmtree('/home/akazemi3/.brain-score/stimuli_on_screen/movshon.FreemanZiemba2013.aperture-public--target8--source4')
            score = model_scores(benchmark=benchmark,layers=layers,prerun=True)


        results = {}
        results['region_dataset']= f'{region}_{dataset}'
        results['mean_score']= score.raw.raw.reset_index('neuroid').values.mean()
        results[param]= param
        print(results)

        if os.path.exists(results_path):
            with open(results_path, 'a', newline='') as new_row:
                writer = DictWriter(new_row, fieldnames=headers)
                writer.writerow(results)
                new_row.close()    

        else:
            with open(results_path, 'w') as first_row:
                writer = DictWriter(first_row, fieldnames = headers)
                writer.writeheader()
                writer.writerows(results)  


                
# Local Data
activations_model = PytorchWrapper(model=model, 
                            preprocessing=PreprocessGS,
                            identifier=f'Model-Object2Vec-{param}')

benchmark = Object2VecEncoderBenchmark(
    data_dir='/data/shared/datasets/object2vec',
    regions=local_data_dict['Object2Vec'],   
    regression='lin')

model_scores = LayerScores(model_identifier=activations_model.identifier,
                       activations_model=activations_model,
                       visual_degrees=8)

score = model_scores(benchmark=benchmark,
                 layers=layers,
                 prerun=True)

df = score.raw.raw.reset_index('neuroid').to_dataframe(name='scores')
for dataset, regions in local_data_dict.items():
    for region in regions:
        results = {}
        results['region_dataset']=f'{region}_{dataset}'
        df = score.raw.raw.reset_index('neuroid').to_dataframe(name='scores')
        results['mean_score']=df[df['region'] == region]['scores'].mean()
        results[param]=param
        print(results)  

        if os.path.exists(results_path):
            with open(results_path, 'a', newline='') as new_row:
                writer = DictWriter(new_row, fieldnames=headers)
                writer.writerow(results)
                new_row.close()    

        else:
            with open(results_path, 'w') as first_row:
                writer = DictWriter(first_row, fieldnames = headers)
                writer.writeheader()
                writer.writerows(results)



