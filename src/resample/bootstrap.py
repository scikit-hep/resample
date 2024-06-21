"""
Bootstrap resampling tools.

Compute estimator bias, variance, confidence intervals with bootstrap resampling.

Several forms of bootstrapping on N-dimensional data are supported: ordinary, balanced,
extended, parametric, and stratified sampling, see :func:`resample` for details.
Parametric bootstrapping fits a user-specified distribution to the data and samples
from the parametric distribution. The distributions are taken from scipy.stats.

Confidence intervals can be computed with the ordinary percentile method and with the
more efficient BCa method, see :func:`confidence_interval` for details.
"""

__all__ = [
    "resample",
    "bootstrap",
    "variance",
    "covariance",
    "confidence_interval",
]

from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from . import _util
from .empirical import quantile_function_gen
from .jackknife import jackknife


def resample(
    sample: "ArrayLike",
    *args: "ArrayLike",
    size: int = 100,
    method: str = "balanced",
    strata: Optional["ArrayLike"] = None,
    random_state: Optional[Union[np.random.Generator, int]] = None,
) -> Generator[np.ndarray, None, None]:
    """
    Return generator of bootstrap samples.

    Parameters
    ----------
    sample : array-like
        Original sample.
    *args : array-like
        Optional additional arrays of the same length to resample.
    size : int, optional
        Number of bootstrap samples to generate. Default is 100.
    method : str or None, optional
        How to generate bootstrap samples. Supported are 'ordinary', 'balanced',
        'extended', or a distribution name for a parametric bootstrap.
        Default is 'balanced'. Supported distribution names: 'normal' (also:
        'gaussian', 'norm'), 'student' (also: 't'), 'laplace', 'logistic',
        'F' (also: 'f'), 'beta', 'gamma', 'log-normal' (also: 'lognorm',
        'log-gaussian'), 'inverse-gaussian' (also: 'invgauss'), 'pareto', 'poisson'.
    strata : array-like, optional
        Stratification labels. Must have the same shape as `sample`. Default is None.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Yields
    ------
    ndarray
        Bootstrap sample.

    Examples
    --------
    Compute uncertainty of arithmetic mean.

    >>> from resample.bootstrap import resample
    >>> import numpy as np
    >>> x = np.arange(10)
    >>> fx = np.mean(x)
    >>> fb = []
    >>> for b in resample(x, size=10000, random_state=1):
    ...     fb.append(np.mean(b))
    >>> print(f"f(x) = {fx:.1f} +/- {np.std(fb):.1f}")
    f(x) = 4.5 +/- 0.9

    Compute uncertainty of function applied to multivariate data.

    >>> from resample.bootstrap import resample
    >>> import numpy as np
    >>> x = np.arange(10)
    >>> y = np.arange(10, 20)
    >>> fx = np.mean((x, y))
    >>> fb = []
    >>> for bx, by in resample(x, y, size=10000, random_state=1):
    ...     fb.append(np.mean((bx, by)))
    >>> print(f"f(x, y) = {fx:.1f} +/- {np.std(fb):.1f}")
    f(x, y) = 9.5 +/- 0.9

    Notes
    -----
    Balanced vs. ordinary bootstrap:

    The balanced bootstrap produces more accurate results for the same number of
    bootstrap samples than the ordinary bootstrap, but needs to allocate memory for
    B integers, where B is the number of bootstrap samples. Since values of B larger
    than 10000 are rarely needed, this is usually not an issue.

    Non-parametric vs. parametric bootstrap:

    If you know that the data follow a particular parametric distribution, it is
    better to sample from this parametric distribution, but in most cases it is
    sufficient and more convenient to do a non-parametric bootstrap (using "balanced",
    "ordinary", "extended"). The parametric bootstrap is essential for estimators
    sensitive to the tails of a distribution (for example, a quantile close to 0 or 1).
    In this case, only a parametric bootstrap will give reasonable answers, since the
    non-parametric bootstrap cannot include rare events in the tail if the original
    sample did not have them.

    Extended bootstrap:

    In particle physics and perhaps also in other fields, estimators are used which are
    that are a function of both the size and shape of a sample (for example, fit of a
    peak over smooth background to the mass distribution of decay candidates). In this
    case, the normal bootstrap (parametric or non-parametric) is not correct, since the
    sample size is kept constant. For such cases, one needs the "extended" bootstrap.
    The name alludes to the so-called extended maximum-likelihood (EML) method in
    particle physics. Estimates obtained with the EML need to be bootstrapped with the
    "extended" bootstrap.

    Stratification:

    If the sample consists of several distinct classes, stratification
    ensures that the relative proportions of each class are maintained in each
    replicated sample. This is a stricter constraint than that offered by the
    balanced bootstrap, which only guarantees that classes have the original
    proportions over all replicates, but not within each one replicate.

    """
    sample_np = np.atleast_1d(sample)
    n_sample = len(sample_np)
    args_np: List[np.ndarray] = []

    if args:
        if not isinstance(args[0], Collection):
            import warnings

            warnings.warn(
                "Calling resample with positional instead of keyword parameters is "
                "deprecated",
                FutureWarning,
            )
            kwargs: Dict[str, Any] = {
                "size": size,
                "method": method,
                "strata": strata,
                "random_state": random_state,
            }
            if len(args) > len(kwargs):
                raise ValueError("too many arguments")
            for key, val in zip(kwargs, args):
                kwargs[key] = val
            size = kwargs["size"]
            method = kwargs["method"]
            strata = kwargs["strata"]
            random_state = kwargs["random_state"]
            del args
        else:
            args_np.append(sample_np)
            for arg in args:
                arg = np.atleast_1d(arg)
                n_arg = len(arg)
                if n_arg != n_sample:
                    raise ValueError(
                        f"extra argument has wrong length {n_arg} != {n_sample}"
                    )
                args_np.append(arg)

    rng = _util.normalize_rng(random_state)

    if strata is not None:
        strata_np = np.atleast_1d(strata)
        if args_np:
            raise ValueError("Stratified resampling only works with one sample array")
        if len(strata_np) != n_sample:
            raise ValueError("a and strata must have the same shape")
        return _resample_stratified(sample_np, size, method, strata_np, rng)

    if method == "balanced":
        if args_np:
            return _resample_balanced_n(args_np, size, rng)
        else:
            return _resample_balanced_1(sample_np, size, rng)
    if method == "ordinary":
        if args_np:
            return _resample_ordinary_n(args_np, size, rng)
        else:
            return _resample_ordinary_1(sample_np, size, rng)
    if method == "extended":
        if args_np:
            return _resample_extended_n(args_np, size, rng)
        else:
            return _resample_extended_1(sample_np, size, rng)

    if args_np:
        raise ValueError("Parametric resampling only works with one sample array")

    dist = {
        # put aliases here
        "gaussian": stats.norm,
        "normal": stats.norm,
        "log-normal": stats.lognorm,
        "log-gaussian": stats.lognorm,
        "inverse-gaussian": stats.invgauss,
        "student": stats.t,
    }.get(method)

    # fallback to scipy.stats name
    if dist is None:
        try:
            dist = getattr(stats, method.lower())
        except AttributeError:
            raise ValueError(f"Invalid family: '{method}'")

    if sample_np.ndim > 1:
        if dist != stats.norm:
            raise ValueError(f"family '{method}' only supports 1D samples")
        dist = stats.multivariate_normal
        if sample_np.ndim > 2:
            raise ValueError("multivariate normal only works with 2D samples")

    return _resample_parametric(sample_np, size, dist, rng)


def bootstrap(
    fn: Callable[..., np.ndarray],
    sample: "ArrayLike",
    *args: "ArrayLike",
    **kwargs: Any,
) -> np.ndarray:
    """
    Calculate function values from bootstrap samples.

    This is equivalent to ``numpy.array([fn(b) for b in resample(sample)])`` and
    implemented for convenience.

    Parameters
    ----------
    fn : Callable
        Bootstrap samples are passed to this function.
    sample : array-like
        Original sample.
    *args : array-like
        Optional additional arrays of the same length to resample.
    **kwargs
        Keywords are forwarded to :func:`resample`.

    Returns
    -------
    np.array
        Results of `fn` applied to each bootstrap sample.

    Examples
    --------
    >>> from resample.bootstrap import bootstrap
    >>> import numpy as np
    >>> x = np.arange(10)
    >>> fx = np.mean(x)
    >>> fb = bootstrap(np.mean, x, size=10000, random_state=1)
    >>> print(f"f(x) = {fx:.1f} +/- {np.std(fb):.1f}")
    f(x) = 4.5 +/- 0.9

    """
    gen = resample(sample, *args, **kwargs)
    if args:
        return np.array([fn(*b) for b in gen])
    return np.array([fn(x) for x in gen])


def variance(
    fn: Callable[..., np.ndarray],
    sample: "ArrayLike",
    *args: "ArrayLike",
    **kwargs: Any,
) -> np.ndarray:
    """
    Calculate bootstrap estimate of variance.

    If the function returns a vector, the variance is computed elementwise.

    Parameters
    ----------
    fn : callable
        Estimator. Can be any mapping ℝⁿ → ℝᵏ, where n is the sample size
        and k is the length of the output array.
    sample : array-like
        Original sample.
    *args : array-like
        Optional additional arrays of the same length to resample.
    **kwargs
        Keyword arguments forwarded to :func:`resample`.

    Returns
    -------
    ndarray
        Bootstrap estimate of variance.

    Examples
    --------
    Compute variance of arithmetic mean.

    >>> from resample.bootstrap import variance
    >>> import numpy as np
    >>> x = np.arange(10)
    >>> v = variance(np.mean, x, size=10000, random_state=1)
    >>> f"{v:.1f}"
    '0.8'

    """
    thetas = bootstrap(fn, sample, *args, **kwargs)
    return np.var(thetas, ddof=1, axis=0)


def covariance(
    fn: Callable[..., np.ndarray],
    sample: "ArrayLike",
    *args: "ArrayLike",
    **kwargs: Any,
) -> np.ndarray:
    """
    Calculate bootstrap estimate of covariance.

    Parameters
    ----------
    fn : callable
        Estimator. Can be any mapping ℝⁿ → ℝᵏ, where n is the sample size
        and k is the length of the output array.
    sample : array-like
        Original sample.
    *args : array-like
        Optional additional arrays of the same length to resample.
    **kwargs
        Keyword arguments forwarded to :func:`resample`.

    Returns
    -------
    ndarray
        Bootstrap estimate of covariance. In general, this is a matrix, but if the
        function maps to a scalar, it is scalar as well.

    Examples
    --------
    Compute covariance of sample mean and sample variance.

    >>> from resample.bootstrap import variance
    >>> import numpy as np
    >>> x = np.arange(10)
    >>> def fn(x):
    ...     return np.mean(x), np.var(x)
    >>> np.round(covariance(fn, x, size=10000, random_state=1), 1)
    array([[0.8, 0. ],
           [0. , 5.5]])

    """
    thetas = bootstrap(fn, sample, *args, **kwargs)
    return np.cov(thetas, rowvar=False, ddof=1)


def confidence_interval(
    fn: Callable[..., np.ndarray],
    sample: "ArrayLike",
    *args: "ArrayLike",
    cl: float = 0.95,
    ci_method: str = "bca",
    **kwargs: Any,
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence intervals.

    Parameters
    ----------
    fn : callable
        Function to be bootstrapped.
    sample : array-like
        Original sample.
    *args : array-like
        Optional additional arrays of the same length to resample.
    cl : float, default : 0.95
        Confidence level. Asymptotically, this is the probability that the interval
        contains the true value.
    ci_method : str, {'bca', 'percentile'}, optional
        Confidence interval method. Default is 'bca'. See notes for details.
    **kwargs
        Keyword arguments forwarded to :func:`resample`.

    Returns
    -------
    (float, float)
        Upper and lower confidence limits.

    Examples
    --------
    Compute confidence interval for arithmetic mean.

    >>> from resample.bootstrap import confidence_interval
    >>> import numpy as np
    >>> x = np.arange(10)
    >>> a, b = confidence_interval(np.mean, x, size=10000, random_state=1)
    >>> f"{a:.1f} to {b:.1f}"
    '2.6 to 6.2'

    Notes
    -----
    Both the 'percentile' and 'bca' methods produce intervals that are invariant to
    monotonic transformations of the data values, a desirable and consistent property.

    The 'percentile' method is straightforward and useful as a fallback. The 'bca'
    method is 2nd order accurate (to O(1/n) where n is the sample size) and generally
    preferred. It computes a jackknife estimate in addition to the bootstrap, which
    increases the number of function evaluations in a direct comparison to
    'percentile'. However the increase in accuracy should compensate for this, with the
    result that less bootstrap replicas are needed overall to achieve the same accuracy.

    """
    if args and not isinstance(args[0], Collection):
        import warnings

        warnings.warn(
            "Calling confidence_interval with positional instead of keyword "
            "arguments is deprecated",
            FutureWarning,
        )

        if len(args) == 1:
            (cl,) = args
        elif len(args) == 2:
            cl, ci_method = args
        else:
            raise ValueError("too many arguments")
        args = ()

    if not 0 < cl < 1:
        raise ValueError("cl must be between zero and one")

    thetas = bootstrap(fn, sample, *args, **kwargs)
    alpha = 1 - cl

    if ci_method == "percentile":
        return _confidence_interval_percentile(thetas, alpha / 2)

    if ci_method == "bca":
        theta = fn(sample, *args)
        j_thetas = jackknife(fn, sample, *args)
        return _confidence_interval_bca(theta, thetas, j_thetas, alpha / 2)

    raise ValueError(
        f"ci_method must be 'percentile' or 'bca', but '{ci_method}' was supplied"
    )


def _resample_stratified(
    sample: np.ndarray,
    size: int,
    method: str,
    strata: np.ndarray,
    rng: np.random.Generator,
) -> Generator[np.ndarray, None, None]:
    # call resample on sub-samples and merge the replicates
    sub_samples = [sample[strata == x] for x in np.unique(strata)]
    for sub_replicates in zip(
        *[resample(s, size=size, method=method, random_state=rng) for s in sub_samples]
    ):
        yield np.concatenate(sub_replicates, axis=0)


def _resample_ordinary_1(
    sample: np.ndarray, size: int, rng: np.random.Generator
) -> Generator[np.ndarray, None, None]:
    # i.i.d. sampling from empirical cumulative distribution of sample
    n = len(sample)
    for _ in range(size):
        yield rng.choice(sample, size=n, replace=True)


def _resample_ordinary_n(
    samples: List[np.ndarray], size: int, rng: np.random.Generator
) -> Generator[np.ndarray, None, None]:
    n = len(samples[0])
    indices = np.arange(n)
    for _ in range(size):
        m = rng.choice(indices, size=n, replace=True)
        yield tuple(s[m] for s in samples)


def _resample_balanced_1(
    sample: np.ndarray, size: int, rng: np.random.Generator
) -> Generator[np.ndarray, None, None]:
    # effectively computes a random permutation of `size` concatenated
    # copies of `sample` and returns `size` equal chunks of that
    n = len(sample)
    indices = rng.permutation(n * size)
    for i in range(size):
        m = indices[i * n : (i + 1) * n] % n
        yield sample[m]


def _resample_balanced_n(
    samples: List[np.ndarray], size: int, rng: np.random.Generator
) -> Generator[np.ndarray, None, None]:
    n = len(samples[0])
    indices = rng.permutation(n * size)
    for i in range(size):
        m = indices[i * n : (i + 1) * n] % n
        yield tuple(s[m] for s in samples)


def _resample_extended_1(
    sample: np.ndarray, size: int, rng: np.random.Generator
) -> Generator[np.ndarray, None, None]:
    # randomly generates the sample size from a Poisson distribution
    n = len(sample)
    for i in range(size):
        k = rng.poisson(1, size=n)
        yield np.repeat(sample, k, axis=0)


def _resample_extended_n(
    samples: List[np.ndarray], size: int, rng: np.random.Generator
) -> Generator[np.ndarray, None, None]:
    n = len(samples[0])
    for i in range(size):
        k = rng.poisson(1, size=n)
        yield tuple(np.repeat(s, k, axis=0) for s in samples)


def _fit_parametric_family(
    dist: stats.rv_continuous, sample: np.ndarray
) -> Tuple[float, ...]:
    if dist == stats.multivariate_normal:
        # has no fit method...
        return np.mean(sample, axis=0), np.cov(sample.T, ddof=1)

    if dist in {stats.f, stats.beta}:
        fit_kwargs = {"floc": 0, "fscale": 1}
    elif dist in {stats.gamma, stats.lognorm, stats.invgauss, stats.pareto}:
        fit_kwargs = {"floc": 0}
    else:
        fit_kwargs = {}

    return dist.fit(sample, **fit_kwargs)  # type: ignore


def _resample_parametric(
    sample: np.ndarray, size: int, dist: stats.rv_continuous, rng: np.random.Generator
) -> Generator[np.ndarray, None, None]:
    n = len(sample)

    # fit parameters by maximum likelihood and sample from that
    if dist == stats.poisson:
        # - poisson has no fit method and there is no scale parameter
        # - random number generation for poisson distribution in scipy seems to be buggy
        mu = np.mean(sample)
        for _ in range(size):
            yield rng.poisson(mu, size=n)
    else:
        args = _fit_parametric_family(dist, sample)
        dist = dist(*args)
        for _ in range(size):
            yield dist.rvs(size=n, random_state=rng)


def _confidence_interval_percentile(
    thetas: np.ndarray, alpha_half: float
) -> Tuple[float, float]:
    quant = quantile_function_gen(thetas)
    return quant(alpha_half), quant(1 - alpha_half)


def _confidence_interval_bca(
    theta: float, thetas: np.ndarray, j_thetas: np.ndarray, alpha_half: float
) -> Tuple[float, float]:
    norm = stats.norm

    # bias correction; implementation notes:
    # - if prop_less is zero, z_naught would become -inf;
    #   we set z_naught to zero then (no bias)
    prop_less = np.mean(thetas < theta)  # proportion of replicates less than obs
    z_naught = norm.ppf(prop_less) if prop_less > 0 else 0.0

    # acceleration; implementation notes:
    # - np.mean returns float even if j_thetas are int,
    #   must convert type explicity to make -= operator work
    # - it is possible that all j_thetas are zero, it then follows
    #   that den and num are zero; we set acc to zero then (no acceleration)
    j_mean = np.mean(j_thetas)
    j_thetas = j_thetas.astype(j_mean.dtype, copy=False)
    j_thetas -= j_mean
    num = np.sum((-j_thetas) ** 3)
    den = np.sum(j_thetas**2)
    acc = num / (6 * den**1.5) if den > 0 else 0.0

    z_low = z_naught + norm.ppf(alpha_half)
    z_high = z_naught + norm.ppf(1 - alpha_half)

    p_low = norm.cdf(z_naught + z_low / (1 - acc * z_low))
    p_high = norm.cdf(z_naught + z_high / (1 - acc * z_high))

    quant = quantile_function_gen(thetas)
    return quant(p_low), quant(p_high)


def __getattr__(key: str) -> Any:
    for match in ("bias", "bias_corrected"):
        if key == match:
            msg = (
                f"resample.bootstrap.{match} has been removed. The implementation was "
                "discovered to be faulty, and a generic fix is not in sight. "
                "Please use resample.jackknife.bias instead."
            )
            raise NotImplementedError(msg)
    raise AttributeError
