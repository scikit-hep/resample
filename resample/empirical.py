"""
Empirical functions
===================

Empirical functions based on a data sample instead of a parameteric density function,
like the empirical CDF. Implemented here are mostly tools used internally.
"""
from typing import Callable, Sequence, Union

import numpy as np

from resample.jackknife import jackknife


def cdf_gen(sample: Union[Sequence, np.ndarray]) -> Callable:
    """
    Return the empirical distribution function for the given sample.

    Parameters
    ----------
    sample : array-like
        Sample.

    Returns
    -------
    callable
        Empirical distribution function.
    """
    sample_np = np.sort(sample)
    n = len(sample_np)
    return lambda x: np.searchsorted(sample_np, x, side="right", sorter=None) / n


def quantile_function_gen(sample: Union[Sequence, np.ndarray]) -> Callable:
    """
    Return the empirical quantile function for the given sample.

    Parameters
    ----------
    sample : array-like
        Sample.

    Returns
    -------
    callable
        Empirical quantile function.
    """
    from typing import Sequence

    sample_np = np.sort(sample)
    n = len(sample_np)

    def quant(p: Sequence) -> np.ndarray:
        ndim = np.ndim(p)  # must come before atleast_1d
        p_np = np.atleast_1d(p)
        result = np.empty(len(p_np))
        valid = (0 <= p_np) & (p_np <= 1)
        idx = np.maximum(np.ceil(p_np[valid] * n).astype(int) - 1, 0)
        result[valid] = sample_np[idx]
        result[~valid] = np.nan
        if ndim == 0:
            return result[0]
        return result

    return quant


def influence(fn: Callable, sample: Sequence) -> np.ndarray:
    """
    Calculate the empirical influence function for a given sample and estimator.

    Parameters
    ----------
    fn : callable
        Estimator. Can be any mapping ℝⁿ → ℝᵏ, where n is the sample size
        and k is the length of the output array.
    sample : array-like
        Sample. Must be one-dimensional.

    Returns
    -------
    ndarray
        Empirical influence values.
    """
    n = len(sample)
    return (n - 1) * (fn(sample) - jackknife(fn, sample))


del Callable
del Sequence
del Union
