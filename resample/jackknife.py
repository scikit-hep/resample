"""
Jackknife resampling.
"""

from typing import Callable, Sequence
import numba as nb
import numpy as np


def resample(sample: Sequence) -> np.ndarray:
    """
    Generator of jackknife'd samples.

    Parameters
    ----------
    sample: array-like
        Sample. If the sequence is multi-dimensional, the first dimension must
        walk over i.i.d. observations.

    Yields
    ------
    ndarray
        Array with same shape and type as input, but with the size of the first
        dimension reduced by one.

    Notes
    -----
    To increase performance, the resampled array is updated on each iteration *in
    place*. If resampled arrays are to be stored, copies have to be made explicitly,
    e.g.:

    >>> from resample.jackknife import resample
    >>> r = []
    >>> for x in resample((1, 2, 3)):
    ...     r.append(x.copy())
    >>> print(r)
    [array([2, 3]), array([1, 3]), array([1, 2])]

    See Also
    --------
    resample.bootstrap.resample
    resample.jackknife.jackknife
    """
    return _resample(np.atleast_1d(sample))


def jackknife(fn: Callable, sample: np.ndarray) -> np.ndarray:
    """
    Calculate jackknife estimates for a given sample and estimator.

    The jackknife is a linear approximation to the bootstrap. In contrast to the
    bootstrap it is deterministic and does not use random numbers. The caveat is the
    computational cost of the jackknife, which is O(n²) for n observations, compared
    to O(n x k) for k bootstrap replicates. For large samples, the bootstrap is more
    efficient.

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
        Jackknife samples.
    """
    return np.asarray([fn(x) for x in resample(sample)])


def bias(fn: Callable, sample: Sequence) -> np.ndarray:
    """
    Calculate jackknife estimate of bias.

    The bias estimate is accurate to O(1/n), where n is the number of samples.
    If the bias is exactly O(1/n), then the estimate is exact.

    Wikipedia:
    https://en.wikipedia.org/wiki/Jackknife_resampling

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
        Jackknife estimate of bias.
    """
    n = len(sample)
    theta = fn(sample)
    mean_theta = np.mean(jackknife(fn, sample), axis=0)
    return (n - 1) * (mean_theta - theta)


def bias_corrected(fn: Callable, sample: Sequence) -> np.ndarray:
    """
    Calculates bias-corrected estimate of the function with the jackknife.

    Removes a bias of O(1/n), where n is the sample size, leaving bias of
    order O(1/n²). If the original function has a bias of exactly O(1/n),
    the corrected result is now unbiased.

    Wikipedia:
    https://en.wikipedia.org/wiki/Jackknife_resampling

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
        Estimate with O(1/n) bias removed.
    """
    n = len(sample)
    theta = fn(sample)
    mean_theta = np.mean(jackknife(fn, sample), axis=0)
    return n * theta - (n - 1) * mean_theta


def variance(fn: Callable, sample: Sequence) -> np.ndarray:
    """
    Calculate jackknife estimate of variance.

    Wikipedia:
    https://en.wikipedia.org/wiki/Jackknife_resampling

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
        Jackknife estimate of variance.
    """
    # formula is (n - 1) / n * sum((fj - mean(fj)) ** 2)
    #   = np.var(fj, ddof=0) * (n - 1)
    thetas = jackknife(fn, sample)
    n = len(sample)
    return (n - 1) * np.var(thetas, ddof=0, axis=0)


@nb.njit
def _resample(sample: np.ndarray) -> np.ndarray:
    """
    Numba implementation of jackknife resampling.
    """
    n = len(sample)
    x = np.empty((n - 1, *sample.shape[1:]), dtype=sample.dtype)
    for i in range(n - 1):
        x[i] = sample[1 + i]
    # yield x0
    yield x.view(x.dtype)  # must return view to avoid a numba life-time bug

    # update of x needs to change values only up to i
    # for a = [0, 1, 2, 3]
    # x0 = [1, 2, 3] (yielded above)
    # x1 = [0, 2, 3]
    # x2 = [0, 1, 3]
    # x3 = [0, 1, 2]
    for i in range(1, n):
        for j in range(i):
            x[j] = sample[j]
        yield x.view(x.dtype)  # must return view to avoid a numba life-time bug
