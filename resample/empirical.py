"""
Various empirical functions.

Empirical means they are based on the original data sample instead of a parameteric
density function.
"""
from typing import Callable, Sequence
import numpy as np
from scipy.interpolate import interp1d as _interp1d
from resample.jackknife import jackknife as _jackknife


def cdf_gen(sample: np.ndarray) -> Callable:
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


def quantile_gen(sample: np.ndarray) -> Callable:
    """
    Return an empirical quantile function for the given sample.

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
    prob = np.arange(1, n + 1, dtype=float) / n

    def quant(p):
        if not 0 <= p <= 1:
            raise ValueError("Argument must be between zero and one")
        if p < 1 / n:
            # TODO: How to handle this case?
            return sample[0]
        return _interp1d(prob, sample)(p)

    return quant


def influence(fn: Callable, sample: Sequence) -> np.ndarray:
    """
    Calculate the empirical influence function for a given sample and estimator
    using the jackknife method.

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
    return (n - 1) * (fn(sample) - _jackknife(fn, sample))
