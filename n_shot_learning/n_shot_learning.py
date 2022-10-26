from __future__ import annotations

import sys
sys.path.append('/home/akazemi3/Desktop/MB_Lab_Project/')
from typing import List, Dict
import os
import torchvision
import torch 
import numpy as np
import logging
import os.path
from result_caching import store_dict
from sklearn.metrics import top_k_accuracy_score, label_ranking_average_precision_score, log_loss,confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.special import softmax
import pandas as pd
from tqdm import tqdm
from model_tools.utils import fullname
from Tools.ImageProcessing import *
from Hand_Engineered.Call_Model import CallModel
from Hand_Engineered.Activations_Extractor import Activations
from unittest.mock import patch
from model_tools.activations import PytorchWrapper
os.environ['RESULTCACHING_DISABLE'] = '1'
from model_tools.activations.pca import LayerPCA
from sklearn.decomposition import PCA


class NShotLearningBase:

    def __init__(self,classifier,
                 n_cats=80, n_train=(10,50), n_test=50,
                 n_repeats=10, identifier='best_ints_model',stimuli_identifier='object2vec'):
        assert classifier in ['linear', 'prototype', 'maxmargin']
        self._logger = logging.getLogger(fullname(self))
        self._classifier = classifier
        self._n_cats = n_cats
        self._n_train = n_train
        self._n_test = n_test
        self._n_repeats = n_repeats
        self.identifier = identifier
        self._stimuli_identifier = stimuli_identifier
        self._layer_performance_statistics = {}

    def fit(self, layers, model_name):
        self._layer_performance_statistics = self._fit(identifier=self.identifier,
                                                       stimuli_identifier=self._stimuli_identifier,
                                                       classifier=self._classifier,
                                                       layers=layers,
                                                       model_name=model_name)

    def as_df(self):
        df = pd.DataFrame()
        for layer, statistics in self._layer_performance_statistics.items():
            for statistic in statistics:
                statistic['layer'] = layer
            df = df.append(statistics, ignore_index=True)
        return df

    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    def _fit(self, identifier, stimuli_identifier, classifier, layers, model_name):
        n_samples = max(self._n_train) + self._n_test
        cat_paths = self.get_image_paths(self._n_cats, n_samples)


        # Compute classification statistics for every layer individually to save on memory.
        # This is more inefficient because we run images through the network several times,
        # but it is a more scalable approach when using many images and large layers.
        layer_performance_statistics = {}
        
        self._logger.debug('Retrieving activations')
        image_list = [path for cat in cat_paths for path in cat]
        cat_labels = [cat[0].split('/')[-2] for cat in cat_paths]
            

        k = 100
        if len(image_list) > k:
            b = 0
            
            activations_list = []
            while b+k<len(image_list):
                image_batch = image_list[b:b+k]
                b+=k
                activations = get_activations(model_name,layers,image_batch,identifier)
                activations_list.append(activations)
            image_batch = image_list[b:]
            activations = get_activations(model_name,layers,image_batch,identifier)
            activations_list.append(activations)            
        
        else:
            activations = get_activations(model_name,layers,image_list,identifier)
        
        for layer in layers:

            layer_activations = np.concatenate([i[layer] for i in activations_list]) if len(image_list) > k else activations[layer] 
            layer_activations = layer_activations.reshape(self._n_cats, n_samples, -1)

            self._logger.debug('Training/evaluating classifiers')
            performance_statistics = []

            sample_orders = np.arange(n_samples)
            for i_repeat in tqdm(range(self._n_repeats), desc='repeat'):
                np.random.seed(i_repeat)
                np.random.shuffle(sample_orders)

                X_test = layer_activations[:, sample_orders[-self._n_test:], :]
                y_test = np.ones((self._n_cats, self._n_test), dtype=int) * \
                         np.arange(self._n_cats).reshape(-1, 1)
                X_test = X_test.reshape(-1, X_test.shape[-1])
                print('X_test shape: ',X_test.shape)
                y_test = y_test.reshape(-1)
                
                cm_sum = np.zeros((len(cat_labels),len(cat_labels)))
                for n_train in self._n_train:
                    X_train = layer_activations[:, sample_orders[:n_train], :]
                    y_train = np.ones((self._n_cats, n_train), dtype=int) * \
                              np.arange(self._n_cats).reshape(-1, 1)
                    X_train = X_train.reshape(-1, X_train.shape[-1])
                    print('X_train: ',X_train.shape)
                    y_train = y_train.reshape(-1)
                    
                    pca = PCA(n_components=100)
                    pca.fit(X_train)
                    performance, cm = self.classifier_performance(pca.transform(X_train), y_train, pca.transform(X_test), y_test)
                    print('pca x_train shape: ',pca.transform(X_train).shape)
                    print('pca x_test shape: ',pca.transform(X_test).shape)
                    
                    #performance, cm = self.classifier_performance(X_train, y_train, X_test, y_test)

                    performance['n_train'] = n_train
                    performance['i_repeat'] = i_repeat
                    performance_statistics.append(performance)
                    
                    if n_train == max(self._n_train):
                        if cm is not None:
                            cm_sum += cm
                    

            if cm is not None:
                cm_sum = pd.DataFrame(cm_sum, index=cat_labels, columns=cat_labels)
                cm_sum.to_csv(f'/home/akazemi3/Desktop/confusion_matrix.csv')
            
            layer_performance_statistics[layer] = performance_statistics
        
        return layer_performance_statistics


    def classifier_performance(self, X_train, y_train, X_test, y_test) -> Dict[str, float]:
        if self._classifier == 'linear':
            return logistic_performance(X_train, y_train, X_test, y_test)
        elif self._classifier == 'prototype':
            return prototype_performance(X_train, y_train, X_test, y_test)
        elif self._classifier == 'maxmargin':
            return maxmargin_performance(X_train, y_train, X_test, y_test)
        else:
            raise ValueError(f'Unknown classifier {self._classifier}')
    
    # image paths for training images
    def get_image_paths(self, n_cats, n_samples) -> List[List[str]]:
        raise NotImplementedError()


class NShotLearningImageFolder(NShotLearningBase):

    def __init__(self, data_dir, *args, **kwargs):
        super(NShotLearningImageFolder, self).__init__(*args, **kwargs)

        assert os.path.exists(data_dir)
        self.data_dir = data_dir

        cat_paths = []
        cats = os.listdir(data_dir)
        assert len(cats) >= self._n_cats
        cats = cats[:self._n_cats]
        n_samples = max(self._n_train) + self._n_test
        
        for cat in cats:
            cat_dir = os.path.join(data_dir, cat)
            files = os.listdir(cat_dir)
            assert len(files) >= n_samples
            files = files[:n_samples]
            paths = [os.path.join(cat_dir, file) for file in files]
            cat_paths.append(paths)
        self.cat_paths = cat_paths

    def get_image_paths(self, n_cats, n_samples) -> List[List[str]]:
        return self.cat_paths

class NShotLearningObject2Vec(NShotLearningImageFolder):
    
    def __init__(self, data_dir, *args, **kwargs):
        super(NShotLearningObject2Vec, self).__init__(data_dir, *args, **kwargs,
                                                      n_train=(5,), n_test=5,
                                                      stimuli_identifier='object2vec')

def logistic_performance(X_train, y_train, X_test, y_test) -> Dict[str, float]:
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)

        top1 = top_k_accuracy_score(y_test, y_pred, k=1)
        top5 = top_k_accuracy_score(y_test, y_pred, k=5)
        mrr = label_ranking_average_precision_score(label_binarize(y_test, classes=range(y_pred.shape[1])), y_pred)
        ll = -log_loss(y_test, y_pred)
        cm = confusion_matrix(y_test, model.predict(X_test))

        return {'accuracy (top 1)': top1,
                'accuracy (top 5)': top5,
                'MRR': mrr,
                'log likelihood': ll}, cm

def prototype_performance(X_train, y_train, X_test, y_test) -> Dict[str, float]:
        model = NearestCentroidDistances()
        model.fit(X_train, y_train)
        y_pred = model.predict_distances(X_test)
        y_pred = softmax(-y_pred, axis=1)   # Simply to order classes based on distance (i.e. not real probabilities)

        top1 = top_k_accuracy_score(y_test, y_pred, k=1)
        top5 = top_k_accuracy_score(y_test, y_pred, k=5)
        mrr = label_ranking_average_precision_score(label_binarize(y_test, classes=range(y_pred.shape[1])), y_pred)
        print('y pred shape:',y_pred.shape)
        cm = None

        return {'accuracy (top 1)': top1,
                'accuracy (top 5)': top5,
                'MRR': mrr}, cm


def maxmargin_performance(X_train, y_train, X_test, y_test) -> Dict[str, float]:
        model = LinearSVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        top1 = top_k_accuracy_score(y_test, y_pred, k=1)

        return {'accuracy (top 1)': top1}


class NearestCentroidDistances(NearestCentroid):
    def predict_distances(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr')
        distances = pairwise_distances(X, self.centroids_, metric=self.metric)
        return distances


def get_activations(model_name,layers,image_list,model_identifier):
    
    if model_name == 'CNN':
        model = torchvision.models.alexnet(pretrained=False)
        activations_model = PytorchWrapper(model=model, 
                                preprocessing=preprocess_RGB,
                                identifier=model_identifier)
        processed_images = preprocess_RGB(image_list)
          
    elif model_name == 'ints':
        model = CallModel()
        activations_model = PytorchWrapper(model=model, 
                                preprocessing=preprocess_GS,
                                identifier=model_identifier)
        processed_images = preprocess_GS(image_list)
    
    return activations_model.get_activations(processed_images,layers)



def get_n_shot_performance(dataset, data_dir, classifier):

    if dataset == 'imagenet21k':
        return NShotLearningImageFolder(data_dir=data_dir,
                                        classifier=classifier,
                                        stimuli_identifier='imagenet21k')
    elif dataset == 'object2vec':
        return NShotLearningObject2Vec(data_dir=data_dir,
                                       classifier=classifier)


def main(dataset, data_dir, classifier, model_name, layers, debug=False):
    results_path = '/home/akazemi3/Desktop'
    
    n_shot_df = pd.DataFrame()
    # gets cat paths
    n_shot = get_n_shot_performance(dataset, data_dir, classifier)
    n_shot.fit(layers,model_name)
    n_shot_df = n_shot_df.append(n_shot.as_df())
    model = CallModel()
    model_info = pd.DataFrame.from_dict({'model layers':[i for i in model.children()]})

    
    with open(f'{results_path}/{model_name}_{dataset}_{classifier}_{layers}_curv_rand_2conv_2ints_standconv_PCA.csv','w') as f:
        n_shot_df.to_csv(f)
        f.write("\n")
        
 #   with open('f{results_path}/{model_name}_{dataset}_{classifier}_{layers}.csv','a') as f:
        f.write("\n")
        model_info.to_csv(f)
        
        
    
    print('finished running code')
    return 



with patch.dict("os.environ", {"RESULTCACHING_HOME": '/home/akazemi3/Documents/cache_temp'}):
    
    main(dataset='imagenet21k', data_dir='/data/shared/datasets/imagenet21k_sorscher2021', 
                        classifier='prototype',model_name='ints',layers=['output'])
    
    #main(dataset='object2vec',data_dir='/data/shared/datasets/object2vec/stimuli',classifier='prototype',model_name='ints',layers=['output'])  
    
    
    