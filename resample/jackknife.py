"""
Jackknife resampling.
"""
from typing import Callable, Generator, Sequence

import numpy as np


def resample(sample: Sequence, copy: bool = True) -> Generator[np.ndarray, None, None]:
    """
    Generator of jackknifed samples.

    Parameters
    ----------
    sample : array-like
        Sample. If the sequence is multi-dimensional, the first dimension must
        walk over i.i.d. observations.
    copy : bool, optional
        If `True`, return the replicated sample as a copy, otherwise return a view into
        the internal array buffer of the generator. Setting this to `False` avoids
        `len(sample)` copies, which is more efficient, but see notes for caveats.

    Yields
    ------
    ndarray
        Array with same shape and type as input, but with the size of the first
        dimension reduced by one. Replicates are missing one value of the original in
        ascending order, e.g. for a sample (1, 2, 3), one gets (2, 3), (1, 3), (1, 2).

    See Also
    --------
    resample.bootstrap.resample : Generate bootstrap samples.
    resample.jackknife.jackknife : Generate jackknife estimates.

    Notes
    -----
    The generator interally keeps a single array to the replicates, which is updated
    on each iteration of the generator. The safe default is to return copies of this
    internal state. To increase performance, it also possible to return a view into
    the generator state, by setting the `copy=False`. However, this will only produce
    correct results if the generator is called strictly sequentially in a single-
    threaded program and the loop body consumes the view and does not try to store it.
    The following program shows what happens otherwise:

    >>> from resample.jackknife import resample
    >>> r1 = []
    >>> for x in resample((1, 2, 3)): # works as expected
    ...     r1.append(x)
    >>> print(r1)
    [array([2, 3]), array([1, 3]), array([1, 2])]
    >>>
    >>> r2 = []
    >>> for x in resample((1, 2, 3), copy=False):
    ...     r2.append(x) # x is now a view into the same array in memory
    >>> print(r2)
    [array([1, 2]), array([1, 2]), array([1, 2])]
    """
    sample = np.atleast_1d(sample)

    n = len(sample)
    x = sample[1:].copy()
    # yield x0
    yield x.copy() if copy else x

    # update of x needs to change only value at index i
    # for a = [0, 1, 2, 3]
    # x0 = [1, 2, 3] (yielded above)
    # x1 = [0, 2, 3] # override first index
    # x2 = [0, 1, 3] # override second index
    # x3 = [0, 1, 2] # ...
    for i in range(n - 1):
        x[i] = sample[i]
        yield x.copy() if copy else x


def jackknife(fn: Callable, sample: Sequence) -> np.ndarray:
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
        Original sample.

    Returns
    -------
    ndarray
        Jackknife samples.
    """
    return np.asarray([fn(x) for x in resample(sample, copy=False)])


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
        Original sample.

    Returns
    -------
    ndarray
        Jackknife estimate of bias (= expectation of estimator - true value).
    """
    n = len(sample)
    theta = fn(sample)
    mean_theta = np.mean(jackknife(fn, sample), axis=0)
    return (n - 1) * (mean_theta - theta)


def bias_corrected(fn: Callable, sample: Sequence) -> np.ndarray:
    """
    Calculate bias-corrected estimate of the function with the jackknife.

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
        Original sample.

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
        Original sample.

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
