

#from typing import Protocol, Any
from collections.abc import Iterable
from tqdm import tqdm
import numpy as np
import torch
import warnings
from torch import nn
import torch.nn.functional as F
import torch
from sklearn.linear_model import Ridge


def z_score(
    x: torch.Tensor, *, dim: int = 0, unbiased: bool = True, nan_policy: str = "omit"
) -> torch.Tensor:
    if nan_policy == "propagate":
        x_mean = x.mean(dim=dim, keepdim=True)
        x_std = x.std(dim=dim, keepdim=True, unbiased=unbiased)
    elif nan_policy == "omit":
        x_mean = x.nanmean(dim=dim, keepdim=True)
        ddof = 1 if unbiased else 0
        x_std = (
            ((x - x_mean) ** 2).sum(dim=dim, keepdim=True) / (x.shape[dim] - ddof)
        ).sqrt()
    else:
        raise ValueError("x contains NaNs")

    x = (x - x_mean) / x_std
    return x


def center(
    x: torch.Tensor, *, dim: int = 0, nan_policy: str = "omit"
) -> torch.Tensor:
    if nan_policy == "propagate":
        x_mean = x.mean(dim=dim, keepdim=True)
    elif nan_policy == "omit":
        x_mean = x.nanmean(dim=dim, keepdim=True)
    else:
        raise ValueError("x contains NaNs")

    x = (x - x_mean)
    return x



def _helper(
    x: torch.Tensor,
    y: torch.Tensor = None,
    *,
    return_value: str,
    return_diagonal: bool = True,
    unbiased: bool = True,
    nan_policy: str = "omit",
) -> torch.Tensor:
    if x.ndim not in {1, 2, 3}:
        raise ValueError(f"x must have 1, 2 or 3 dimensions (n_dim = {x.ndim})")
    x = x.unsqueeze(1) if x.ndim == 1 else x

    dim_sample_x, dim_feature_x = x.ndim - 2, x.ndim - 1
    n_samples_x = x.shape[dim_sample_x]
    n_features_x = x.shape[dim_feature_x]

    if return_value == "pearson_r":
        x = z_score(x, dim=dim_sample_x, unbiased=unbiased, nan_policy=nan_policy)
    elif return_value == "covariance":
        x = center(x, dim=dim_sample_x, nan_policy=nan_policy)

    if y is not None:
        if y.ndim not in {1, 2, 3}:
            raise ValueError(f"y must have 1, 2 or 3 dimensions (n_dim = {y.ndim})")
        y = y.unsqueeze(1) if y.ndim == 1 else y

        dim_sample_y, dim_feature_y = y.ndim - 2, y.ndim - 1
        n_samples_y = y.shape[dim_sample_y]

        if n_samples_x != n_samples_y:
            raise ValueError(
                f"x and y must have same n_samples (x={n_samples_x}, y={n_samples_y}"
            )

        if return_diagonal:
            n_features_y = y.shape[dim_feature_y]
            if n_features_x != n_features_y:
                raise ValueError(
                    "x and y must have same n_features to return diagonal"
                    f" (x={n_features_x}, y={n_features_y})"
                )

        if return_value == "pearson_r":
            y = z_score(y, dim=dim_sample_y, unbiased=unbiased, nan_policy=nan_policy)
        elif return_value == "covariance":
            y = center(y, dim=dim_sample_y, nan_policy=nan_policy)
    else:
        y = x

    x = torch.matmul(x.transpose(-2, -1), y) / (n_samples_x - 1)
    if return_diagonal:
        x = torch.diagonal(x, dim1=-2, dim2=-1)
    return x.squeeze()


def pearson_r(
    x: torch.Tensor,
    y: torch.Tensor = None,
    *,
    return_diagonal: bool = True,
    unbiased: bool = True,
    nan_policy: str = "omit",
) -> torch.Tensor:
    """Computes Pearson correlation coefficients.
    x and y optionally take a batch dimension (either x or y, or both; in the former case, the pairwise correlations are broadcasted along the batch dimension). If x and y are both specified, pairwise correlations 
    between the columns of x and those of y are computed.
    :param x: a tensor of shape (*, n_samples, n_features) or (n_samples,)
    :param y: an optional tensor of shape (*, n_samples, n_features) or (n_samples,), defaults to None
    :param return_diagonal: when both x and y are specified and have corresponding features (i.e. equal n_features), returns only the (*, n_features) diagonal of the (*, n_features, n_features) pairwise correlation 
    matrix, defaults to True
    :return: Pearson correlation coefficients (*, n_features_x, n_features_y)
    """
    return _helper(
        x=x,
        y=y,
        return_value="pearson_r",
        return_diagonal=return_diagonal,
        unbiased=unbiased,
        nan_policy=nan_policy,
    )


def covariance(
    x: torch.Tensor,
    y: torch.Tensor = None,
    *,
    return_diagonal: bool = True,
    unbiased: bool = True,
    nan_policy: str = "omit",
) -> torch.Tensor:
    """Computes covariance.
    x and y optionally take a batch dimension (either x or y, or both; in the former case, the pairwise covariances are broadcasted along the batch dimension). If x and y are both specified, pairwise covariances between the columns of x and those of y are computed.
    :param x: a tensor of shape (*, n_samples, n_features) or (n_samples,)
    :param y: an optional tensor of shape (*, n_samples, n_features) or (n_samples,), defaults to None
    :param return_diagonal: when both x and y are specified and have corresponding features (i.e. equal n_features), returns only the (*, n_features) diagonal of the (*, n_features, n_features) pairwise covariance matrix, defaults to True
    :return: covariance matrix (*, n_features_x, n_features_y)
    """
    return _helper(
        x=x,
        y=y,
        return_value="covariance",
        return_diagonal=return_diagonal,
        unbiased=unbiased,
        nan_policy=nan_policy,
    )




class LinearRegression:
    # TODO support a batch dimension
    def __init__(self, fit_intercept: bool = True, copy: bool = True):
        self.fit_intercept = fit_intercept
        self.copy_: bool = copy
        self.device_: torch.device = None
        self.n_features_in_: int = None
        self.coef_: torch.Tensor = None
        self.intercept_: torch.Tensor = None
        self.rank_: int = None

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        rcond = None,
        driver = None,
    ) -> None:
        if self.coef_ is not None:
            warnings.warn("Model has already been fit", category=RuntimeWarning)

        self.device_ = x.device
        if self.copy_:
            _x = torch.clone(x)
        else:
            _x = x

        _x = _x.unsqueeze(dim=-1) if _x.ndim == 1 else _x
        y = y.unsqueeze(dim=-1) if y.ndim == 1 else y

        if _x.ndim == 3 and y.ndim == 2:
            y = y.unsqueeze(0)
        n_samples, self.n_features_in_ = _x.shape[-2], _x.shape[-1]
        if y.shape[-2] != n_samples:
            raise ValueError(
                f"number of samples in x and y must be equal (x={n_samples},"
                f" y={y.shape[-2]})"
            )

        if self.fit_intercept:
            x_mean = _x.mean(dim=-2, keepdim=True)
            _x = _x - x_mean
            y_mean = y.mean(dim=-2, keepdim=True)
            y = y - y_mean

        
        print(_x.shape)
        (
            self.coef_,
            _,
            self.rank_,
            _,
        ) = torch.linalg.lstsq(_x + 10 * torch.eye(n=_x.shape[0],m=_x.shape[1]), y, rcond=rcond, driver=driver)
        self.coef_ = self.coef_.transpose(-2, -1)
            
            
        if self.fit_intercept:
            self.intercept_ = y_mean - torch.matmul(
                x_mean, self.coef_.transpose(-2, -1)
            )
        else:
            self.intercept_ = torch.zeros(1, device=self.device_)

    def predict(self, x: torch.Tensor):
        if self.coef_ is None:
            raise RuntimeError("Model has not been fit")
        return torch.matmul(x, self.coef_.transpose_(-2, -1)) + self.intercept_

    def to(self, device: torch.device):
        self.device_ = torch.device(device)
        self.coef_.to(device)
        self.intercept_.to(device)


               


    
    
    
class Regression():
    ...

    def fit(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        ...

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        ...


def create_splits(n: int, *, n_folds: int = 5, shuffle: bool = True): #-> list[npt.NDArray[int]]:
    if shuffle:
        rng = np.random.default_rng(seed=0)
        indices = rng.permutation(n)
    else:
        indices = np.arange(n)

    x = np.array_split(indices, n_folds)
    return x


def regression(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    model: Regression = Ridge(alpha=1),
    indices_train=None,
    indices_test= None,
):
    x = torch.nan_to_num(x)
    
    if indices_train is None and indices_test is not None:
        indices_train = np.setdiff1d(np.arange(x.shape[-2]), np.array(indices_test))
    elif indices_test is None and indices_train is not None:
        indices_test = np.setdiff1d(np.arange(x.shape[-2]), np.array(indices_train))
    elif indices_train is None and indices_test is None:
        indices_train = np.arange(x.shape[-2])
        indices_test = np.arange(x.shape[-2])

    x_train, x_test = x[..., indices_train, :], x[..., indices_test, :]
    y_train, y_test = y[..., indices_train, :], y[..., indices_test, :]

    model.fit(x_train, y_train)

    y_predicted = torch.Tensor(model.predict(x_test))
    return y_test, y_predicted



def regression_cv(x: torch.Tensor,
    y: torch.Tensor,
    model: Regression = Ridge(alpha=1),
    n_folds: int = 8,
    shuffle: bool = True,
):
    y_true, y_predicted = [], []

    splits = create_splits(n=y.shape[-2], n_folds=n_folds, shuffle=shuffle)
    for indices_test in tqdm(splits, desc="split", leave=False):
        y_true_, y_predicted_ = regression(
            model=model,
            x=x,
            y=y,
            indices_test=indices_test,
        )
        y_true.append(y_true_)
        y_predicted.append(y_predicted_)

    return y_true, y_predicted


def regression_cv_concatenated(x: torch.Tensor,
    y: torch.Tensor,
    model: Regression = Ridge(alpha=1),
    n_folds: int = 8,
    shuffle: bool = True,
):
    y_true, y_predicted = regression_cv(
        x=x, y=y, model=model, n_folds=n_folds, shuffle=shuffle
    )
    y_predicted = torch.concat(y_predicted, dim=-2)
    y_true = torch.concat(y_true, dim=-2)
    return y_true, y_predicted