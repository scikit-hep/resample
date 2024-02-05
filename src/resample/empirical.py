"""
Empirical functions.

Empirical functions based on a data sample instead of a parameteric density function,
like the empirical CDF. Implemented here are mostly tools used internally.
"""

__all__ = ["cdf_gen", "quantile_function_gen", "influence"]

from typing import Callable, Union

import numpy as np
from numpy.typing import ArrayLike

from .jackknife import jackknife


def cdf_gen(sample: "ArrayLike") -> Callable[[np.ndarray], np.ndarray]:
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


def quantile_function_gen(
    sample: "ArrayLike",
) -> Callable[[Union[float, "ArrayLike"]], Union[float, np.ndarray]]:
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

    class QuantileFn:
        def __init__(self, sample: "ArrayLike"):
            self._sorted = np.sort(sample, axis=0)

        def __call__(self, p: Union[float, "ArrayLike"]) -> Union[float, np.ndarray]:
            ndim = np.ndim(p)  # must come before atleast_1d
            p = np.atleast_1d(p)
            result = np.empty(len(p))
            valid = (p >= 0) & (p <= 1)
            n = len(self._sorted)
            idx = np.maximum(np.ceil(p[valid] * n).astype(int) - 1, 0)
            result[valid] = self._sorted[idx]
            result[~valid] = np.nan
            if ndim == 0:
                return result[0]
            return result

    return QuantileFn(sample)


def influence(
    fn: Callable[["ArrayLike"], np.ndarray], sample: "ArrayLike"
) -> np.ndarray:
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
    sample = np.atleast_1d(sample)
    n = len(sample)
    return (n - 1) * (fn(sample) - jackknife(fn, sample))
