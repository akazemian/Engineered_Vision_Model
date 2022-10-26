

from abc import ABC, abstractmethod
import logging
from collections import OrderedDict
from typing import Any, Union, Optional, List, Dict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from brainio.stimuli import StimulusSet
from brainio.assemblies import NeuroidAssembly
from model_tools.activations import PytorchWrapper, TensorflowWrapper, TensorflowSlimWrapper, KerasWrapper
#from model_tools.utils import fullname
from result_caching import store_dict

ActivationsModel = Union[PytorchWrapper, TensorflowWrapper, TensorflowSlimWrapper, KerasWrapper]


class ModelActivationsAnalysis(ABC):

    def __init__(self, activations_model: ActivationsModel):
        self._activations_model = activations_model
        #self._logger = logging.getLogger(fullname(self))
        self._layer_results = None
        self._metadata = {}

    @abstractmethod
    def analysis_func(self, assembly: NeuroidAssembly, **kwargs) -> Any:
        """
        This is the only function that needs to be implemented in subclasses.
        Any additional kwargs that this function takes must also be passed to __call__().
        :param assembly: An assembly on which to perform analysis
            (e.g. acitvations or neural recordings in response to stimuli).
        :param kwargs: Additional arguments required for this analysis (e.g. a batch size).
            Any additional kwargs that this function takes must also be passed to __call__().
        """
        pass

    @property
    def results(self) -> Any:
        """
        This property can be overidden in subclasses if you wish to transform the format of the results
        (e.g. merge the dictionary into an xarray DataArray).
        """
        return self._layer_results

    @property
    def identifier(self) -> str:
        return self._activations_model.identifier

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def __call__(self,
                 stimuli: Union[StimulusSet, List[str], List[Path]],
                 layers: List[str],
                 stimuli_identifier: Optional[Union[bool, str]] = None,
                 **kwargs) -> Any:
        """
        Perform an analysis on a model given a set of stimuli.
        :param stimuli: Stimuli to obtain activations from.
        :param layers: Layers to obtain activations from.
        :param stimuli_identifier: Identifier used for stimulus set (used for caching filename).
            Setting to False will prevent activations and analysis results from being cached.
            If caching is desired, either 'stimuli' must be a StimulusSet or 'stimuli_identifier' must be a string.
        :param kwargs: Additional arguments that are passed down to 'self.analysis_func()'.
        :return: Arbitrary analysis data structure.
        """
        if isinstance(stimuli, StimulusSet) and stimuli_identifier is None:
            stimuli_identifier = stimuli.identifier

        if stimuli_identifier:
            self._layer_results = self._run_analysis_stored(identifier=self._activations_model.identifier,
                                                            stimuli=stimuli,
                                                            layers=layers,
                                                            stimuli_identifier=stimuli_identifier,
                                                            analysis_kwargs=kwargs)
            self._metadata['stimuli_identifier'] = stimuli_identifier
        else:
            self._layer_results = self._run_analysis(stimuli=stimuli,
                                                     layers=layers,
                                                     stimuli_identifier=stimuli_identifier,
                                                     **kwargs)

        return self.results

    @store_dict(dict_key='layers', identifier_ignore=['stimuli', 'layers'])
    def _run_analysis_stored(self,
                             identifier: str,
                             stimuli: Union[StimulusSet, List[str], List[Path]],
                             layers: List[str],
                             stimuli_identifier: str,
                             analysis_kwargs: Optional[Dict] = None) -> Dict[str, Any]:
        if analysis_kwargs is None:
            analysis_kwargs = {}
        layer_results = self._run_analysis(stimuli=stimuli,
                                           layers=layers,
                                           stimuli_identifier=stimuli_identifier,
                                           **analysis_kwargs)
        self._metadata['stimuli_identifier'] = stimuli_identifier
        return layer_results

    def _run_analysis(self,
                      stimuli: Union[StimulusSet, List[str], List[Path]],
                      layers: List[str],
                      stimuli_identifier: Optional[Union[bool, str]] = None,
                      **kwargs) -> Dict[str, Any]:
        #self._logger.debug('Obtaining activations')
        layer_activations = self._activations_model(stimuli=stimuli, layers=layers,
                                                    stimuli_identifier=stimuli_identifier)

        #self._logger.debug('Performing analyses')
        layers = np.unique(layer_activations['layer'])
        progress = tqdm(total=len(layers), desc="layer analyses")

        layer_results = OrderedDict()
        for layer in layers:
            activations = layer_activations.sel(layer=layer)
            layer_results[layer] = self.analysis_func(assembly=activations, **kwargs)
            progress.update(1)

        progress.close()
        return layer_results
    
    
    
from typing import Any, Optional

import numpy as np
import xarray as xr
import torch
from sklearn.linear_model import LinearRegression

from brainio.assemblies import NeuroidAssembly
from result_caching import store_xarray
from analysis.eigen_spectrum.utils import IncrementalPCAPytorch, PCAPytorch


class ModelEigspecAnalysis(ModelActivationsAnalysis):

    def analysis_func(self,
                      assembly: NeuroidAssembly,
                      batch_size: Optional[int] = None,
                      device=None) -> Any:
        return get_eigspec(assembly, batch_size, device)

    @property
    def results(self) -> xr.DataArray:
        layer_results, layers = self._layer_results.values(), list(self._layer_results.keys())
        layer_results = xr.concat(layer_results, dim='identifier')
        layer_results['identifier'] = [self.identifier] * len(layers)
        layer_results['layer'] = ('identifier', layers)
        for name, data in self.metadata.items():
            layer_results[name] = ('identifier', [data] * len(layers))
        return layer_results


def get_eigspec(assembly: NeuroidAssembly,
                batch_size: Optional[int] = None,
                device=None) -> xr.DataArray:
    if batch_size is None:
        pca = PCAPytorch(device=device)
        assembly = torch.from_numpy(assembly.values).to(device)
        pca.fit(assembly)
    else:
        pca = IncrementalPCAPytorch(device=device)
        for i in range(0, assembly.shape[0], batch_size):
            assembly_batch = torch.from_numpy(assembly[i:i + batch_size].values).to(device)
            pca.fit_partial(assembly_batch)
    eigspec = pca.explained_variance_

    eigspec = eigspec.cpu().numpy()
    ed = effective_dimensionalities(eigspec)
    eighty_percent_var = x_percent_var(eigspec, x=0.8)
    alpha = powerlaw_exponent(eigspec)

    eigspec = xr.DataArray(eigspec.reshape(1, -1),
                           dims=['identifier', 'eigval_index'],
                           coords={'effective_dimensionality': ('identifier', [ed]),
                                   'eighty_percent_var': ('identifier', [eighty_percent_var]),
                                   'powerlaw_decay_exponent': ('identifier', [alpha]),
                                   'eigval_index': np.arange(1, len(eigspec) + 1)})
    return eigspec


@store_xarray(identifier_ignore=['assembly', 'batch_size', 'device'], combine_fields=[])
def get_eigspec_stored(identifier: str,
                       assembly: NeuroidAssembly,
                       batch_size: Optional[int] = None,
                       device=None) -> xr.DataArray:
    eigspec = get_eigspec(assembly=assembly, batch_size=batch_size, device=device)
    eigspec['identifier'] = [identifier]
    return eigspec


############################################################
#################### Helper functions ######################
############################################################


def effective_dimensionalities(eigspec: np.ndarray) -> float:
    return eigspec.sum() ** 2 / (eigspec ** 2).sum()


def x_percent_var(eigspec: np.ndarray, x: float) -> float:
    assert 0 < x < 1
    i_varx = None
    pvar = eigspec.cumsum() / eigspec.sum()
    for i in range(len(pvar)):
        if pvar[i] >= x:
            i_varx = i + 1
            break
    return i_varx


def powerlaw_exponent(eigspec: np.ndarray) -> float:
    start, end = 0, np.log10(len(eigspec))
    eignum = np.logspace(start, end, num=50).round().astype(int)
    eigspec = eigspec[eignum - 1]
    logeignum = np.log10(eignum)
    logeigspec = np.log10(eigspec)

    # remove infs when eigenvalues are too small
    filter_ = ~np.isinf(logeigspec)
    logeignum = logeignum[filter_]
    logeigspec = logeigspec[filter_]
    linear_fit = LinearRegression().fit(logeignum.reshape(-1, 1), logeigspec)
    alpha = -linear_fit.coef_.item()
    return alpha