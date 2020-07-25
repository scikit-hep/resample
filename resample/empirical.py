"""
Various empirical functions.

Empirical means they are based on the original data sample instead of a parameteric
density function.
"""
from typing import Callable, Sequence

import numpy as np

from resample.jackknife import jackknife


def cdf_gen(sample: Sequence) -> Callable:
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
    sample = np.sort(sample)
    n = len(sample)
    return lambda x: np.searchsorted(sample, x, side="right", sorter=None) / n


def quantile_function_gen(sample: Sequence) -> Callable:
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
    sample = np.sort(sample)
    n = len(sample)

    def quant(p):
        ndim = np.ndim(p)
        p = np.atleast_1d(p)
        result = np.empty(len(p))
        valid = (0 <= p) & (p <= 1)
        idx = np.maximum(np.ceil(p[valid] * n).astype(np.int) - 1, 0)
        result[valid] = sample[idx]
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
