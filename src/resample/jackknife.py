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
is involved, but this can be overcome by fixing the seed for the bootstrap.

The jackknife should be used to estimate the bias, since the bootstrap cannot (easily)
estimate bias. The bootstrap should be preferred when computing the variance.
"""

__all__ = [
    "resample",
    "jackknife",
    "bias",
    "bias_corrected",
    "variance",
    "cross_validation",
]

from typing import Any, Callable, Collection, Generator, List

import numpy as np
from numpy.typing import ArrayLike


def resample(
    sample: "ArrayLike", *args: "ArrayLike", copy: bool = True
) -> Generator[Any, None, None]:
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
        if not isinstance(args[0], Collection):
            import warnings

            warnings.warn(
                "Calling resample with positional instead of keyword arguments is "
                "deprecated",
                FutureWarning,
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


def _resample_1(sample: np.ndarray, copy: bool) -> Generator[np.ndarray, None, None]:
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


def _resample_n(samples: List[np.ndarray], copy: bool) -> Generator[Any, None, None]:
    x = [a[1:].copy() for a in samples]
    yield (xi.copy() for xi in x)
    for i in range(len(samples[0]) - 1):
        for xi, ai in zip(x, samples):
            xi[i] = ai[i]
        yield (xi.copy() for xi in x)


def jackknife(
    fn: Callable[..., np.ndarray],
    sample: "ArrayLike",
    *args: "ArrayLike",
) -> np.ndarray:
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


def bias(
    fn: Callable[..., np.ndarray], sample: "ArrayLike", *args: "ArrayLike"
) -> np.ndarray:
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
    >>> b1 = bias(np.var, x)
    >>> b2 = bias(lambda x: np.var(x, ddof=1), x)
    >>> f"bias of naive sample variance {b1:.1f}, bias of corrected variance {b2:.1f}"
    'bias of naive sample variance -0.9, bias of corrected variance 0.0'

    """
    sample = np.atleast_1d(sample)
    n = len(sample)
    theta = fn(sample)
    mean_theta = np.mean(jackknife(fn, sample, *args), axis=0)
    return (n - 1) * (mean_theta - theta)


def bias_corrected(
    fn: Callable[..., np.ndarray], sample: "ArrayLike", *args: "ArrayLike"
) -> np.ndarray:
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
    >>> v1 = np.var(x)
    >>> v2 = bias_corrected(np.var, x)
    >>> f"naive variance {v1:.1f}, bias-corrected variance {v2:.1f}"
    'naive variance 8.2, bias-corrected variance 9.2'

    """
    sample = np.atleast_1d(sample)
    n = len(sample)
    theta = fn(sample)
    mean_theta = np.mean(jackknife(fn, sample, *args), axis=0)
    return n * theta - (n - 1) * mean_theta


def variance(
    fn: Callable[..., np.ndarray], sample: "ArrayLike", *args: "ArrayLike"
) -> np.ndarray:
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
    >>> v = variance(np.mean, x)
    >>> f"{v:.1f}"
    '0.9'

    """
    # formula is (n - 1) / n * sum((fj - mean(fj)) ** 2)
    #   = np.var(fj, ddof=0) * (n - 1)
    sample = np.atleast_1d(sample)
    thetas = jackknife(fn, sample, *args)
    n = len(sample)
    return (n - 1) * np.var(thetas, ddof=0, axis=0)


def cross_validation(
    predict: Callable[..., float], x: "ArrayLike", y: "ArrayLike", *args: "ArrayLike"
) -> float:
    """
    Calculate mean-squared error of model with leave-one-out-cross-validation.

    Wikipedia:
    https://en.wikipedia.org/wiki/Cross-validation_(statistics)

    Parameters
    ----------
    predict : callable
        Function with the signature (x_in, y_in, x_out, *args). It takes x_in, y_in,
        which are arrays with the same length. x_out should be one element of the x
        array. *args are further optional arguments for the function. The function
        should return the prediction y(x_out).
    x : array-like
        Explanatory variable. Must be an array of shape (N, ...), where N is the number
        of samples.
    y : array-like
        Observations. Must be an array of shape (N, ...).
    *args:
        Optional arguments which are passed unmodified to predict.

    Returns
    -------
    float
        Variance of the difference (y[i] - predict(..., x[i], *args)).

    """
    deltas = []
    for i, (x_in, y_in) in enumerate(resample(x, y, copy=False)):
        yip = predict(x_in, y_in, x[i], *args)
        deltas.append((y[i] - yip))
    return np.var(deltas)  # type:ignore
