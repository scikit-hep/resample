"""
Jackknife resampling tools.

Compute estimator bias and variance with jackknife resampling. The implementation
supports resampling of N-dimensional data. The interface of this module mimics that of
the bootstrap module, so that you can easily switch between bootstrapping and
jackknifing bias and variance of an estimator.

The jackknife is an approximation to the bootstrap, so in general bootstrapping is
preferred, especially when the sample is small. The computational cost of the jackknife
increases quadratically with the sample size, but only linearly for the bootstrap. An
advantage of the jackknife can be the deterministic outcome, since no random sampling
is involved.
"""
import typing as _tp

import numpy as np

_Args = _tp.Any
_ArrayLike = _tp.Collection


def resample(
    sample: _ArrayLike, *args: _Args, copy: bool = True
) -> _tp.Generator[np.ndarray, None, None]:
    """
    Generate jackknifed samples.

    Parameters
    ----------
    sample : array-like
        Sample. If the sequence is multi-dimensional, the first dimension must
        walk over i.i.d. observations.
    *args: array-like
        Optional additional arrays of the same length to resample.
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
    On performance:

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
    sample_np = np.atleast_1d(sample)
    n_sample = len(sample_np)

    args_np = []
    if args:
        if not isinstance(args[0], _tp.Collection):
            import warnings

            from numpy import VisibleDeprecationWarning

            warnings.warn(
                "Calling resample with positional instead of keyword arguments is "
                "deprecated",
                VisibleDeprecationWarning,
            )
            if len(args) == 1:
                (copy,) = args
            else:
                raise ValueError("too many arguments")
        else:
            args_np.append(sample_np)
            for arg in args:
                arg_np = np.atleast_1d(arg)
                n_arg = len(arg_np)
                if n_arg != n_sample:
                    raise ValueError(
                        f"extra argument has wrong length {n_arg} != {n_sample}"
                    )
                args_np.append(arg_np)

    if args_np:
        return _resample_n(args_np, copy)
    return _resample_1(sample_np, copy)


def _resample_1(sample: np.ndarray, copy: bool):
    # yield x0
    x = sample[1:].copy()
    yield x.copy() if copy else x

    # update of x needs to change only value at index i
    # for a = [0, 1, 2, 3]
    # x0 = [1, 2, 3] (yielded above)
    # x1 = [0, 2, 3] # override first index
    # x2 = [0, 1, 3] # override second index
    # x3 = [0, 1, 2] # ...
    for i in range(len(sample) - 1):
        x[i] = sample[i]
        yield x.copy() if copy else x


def _resample_n(samples: _tp.List[np.ndarray], copy: bool):
    x = [a[1:].copy() for a in samples]
    yield (xi.copy() for xi in x)
    for i in range(len(samples[0]) - 1):
        for xi, ai in zip(x, samples):
            xi[i] = ai[i]
        yield (xi.copy() for xi in x)


def jackknife(fn: _tp.Callable, sample: _ArrayLike, *args: _Args) -> np.ndarray:
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
    *args: array-like
        Optional additional arrays of the same length to resample.

    Returns
    -------
    ndarray
        Jackknife samples.

    Examples
    --------
    >>> from resample.jackknife import jackknife
    >>> import numpy as np
    >>> x = np.arange(10)
    >>> fx = np.mean(x)
    >>> fb = jackknife(np.mean, x)
    >>> print(f"f(x) = {fx:.1f} +/- {np.std(fb):.1f}")
    f(x) = 4.5 +/- 0.3
    """
    gen = resample(sample, *args, copy=False)
    if args:
        return np.array([fn(*b) for b in gen])
    return np.asarray([fn(b) for b in gen])


def bias(fn: _tp.Callable, sample: _ArrayLike, *args: _Args) -> np.ndarray:
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
    *args: array-like
        Optional additional arrays of the same length to resample.

    Returns
    -------
    ndarray
        Jackknife estimate of bias (= expectation of estimator - true value).

    Examples
    --------
    Compute bias of numpy.var with and without bias-correction.

    >>> from resample.jackknife import bias
    >>> import numpy as np
    >>> x = np.arange(10)
    >>> round(bias(np.var, x), 1)
    -0.9
    >>> round(bias(lambda x: np.var(x, ddof=1), x), 1)
    0.0
    """
    sample = np.atleast_1d(sample)
    n = len(sample)
    theta = fn(sample)
    mean_theta = np.mean(jackknife(fn, sample, *args), axis=0)
    return (n - 1) * (mean_theta - theta)


def bias_corrected(fn: _tp.Callable, sample: _ArrayLike, *args: _Args) -> np.ndarray:
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
    *args: array-like
        Optional additional arrays of the same length to resample.

    Returns
    -------
    ndarray
        Estimate with O(1/n) bias removed.

    Examples
    --------
    Compute bias-corrected estimate of numpy.var.

    >>> from resample.jackknife import bias_corrected
    >>> import numpy as np
    >>> x = np.arange(10)
    >>> round(np.var(x), 1)
    8.2
    >>> round(bias_corrected(np.var, x), 1)
    9.2
    """
    sample = np.atleast_1d(sample)
    n = len(sample)
    theta = fn(sample)
    mean_theta = np.mean(jackknife(fn, sample, *args), axis=0)
    return n * theta - (n - 1) * mean_theta


def variance(fn: _tp.Callable, sample: _ArrayLike, *args: _Args) -> np.ndarray:
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
    *args: array-like
        Optional additional arrays of the same length to resample.

    Returns
    -------
    ndarray
        Jackknife estimate of variance.

    Examples
    --------
    Compute variance of arithmetic mean.

    >>> from resample.jackknife import variance
    >>> import numpy as np
    >>> x = np.arange(10)
    >>> round(variance(np.mean, x), 1)
    0.9
    """
    # formula is (n - 1) / n * sum((fj - mean(fj)) ** 2)
    #   = np.var(fj, ddof=0) * (n - 1)
    sample = np.atleast_1d(sample)
    thetas = jackknife(fn, sample, *args)
    n = len(sample)
    return (n - 1) * np.var(thetas, ddof=0, axis=0)
