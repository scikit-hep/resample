"""
Bootstrap resampling
====================

Compute estimator bias, variance, confidence intervals with bootstrap resampling.
Several forms of bootstrapping on N-dimensional data are supported: ordinary, balanced,
parametric, and stratified sampling. Parametric bootstrapping fits a user-specified
distribution to the data and samples from the parametric distribution. The distributions
are taken from scipy.stats.

Confidence intervals can be computed with the ordinary percentile method and with the
more efficient BCa method.
"""
import typing as _tp

import numpy as np
from scipy import stats

from resample.empirical import quantile_function_gen
from resample.jackknife import jackknife

_Kwargs = _tp.Any


def resample(
    sample: _tp.Iterable,
    *args: _tp.Tuple[_tp.Any],
    size: int = 100,
    method: str = "balanced",
    strata: _tp.Optional[_tp.Iterable] = None,
    random_state: _tp.Optional[_tp.Union[np.random.Generator, int]] = None,
) -> _tp.Generator[np.ndarray, None, None]:
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
        How to generate bootstrap samples. Supported are 'ordinary', 'balanced', or
        a distribution name for a parametric bootstrap. Default is 'balanced'.
        Supported distribution names: 'normal' (also: 'gaussian', 'norm'),
        'student' (also: 't'), 'laplace', 'logistic', 'F' (also: 'f'),
        'beta', 'gamma', 'log-normal' (also: 'lognorm', 'log-gaussian'),
        'inverse-gaussian' (also: 'invgauss'), 'pareto', 'poisson'.
    strata : array-like, optional
        Stratification labels. Must have the same shape as `sample`. Default is None.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Yields
    ------
    ndarray
        Bootstrap sample.

    Notes
    -----
    If data is not i.i.d. but consists of several distinct classes, stratification
    ensures that the relative proportions of each class are maintained in each
    replicated sample. This is a stricter constraint than that offered by the
    balanced bootstrap, which only guarantees that classes have the original
    proportions over all replicates.
    """
    sample_np = np.atleast_1d(sample)
    n_sample = len(sample_np)
    args_np = []
    if args:
        if not isinstance(args[0], _tp.Iterable):
            import warnings

            from numpy import VisibleDeprecationWarning

            warnings.warn(
                "Calling resample with positional instead of keyword arguments is "
                "deprecated",
                VisibleDeprecationWarning,
            )
            if len(args) == 1:
                (size,) = args
            elif len(args) == 2:
                size, method = args
            elif len(args) == 3:
                size, method, strata = args
            elif len(args) == 4:
                size, method, strata, random_state = args
            else:
                raise ValueError("too many arguments")
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

    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

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
    }.get(method, None)

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


def bootstrap(fn: _tp.Callable, sample: _tp.Iterable, **kwargs: _Kwargs) -> np.ndarray:
    """
    Calculate function values from bootstrap samples.

    Parameters
    ----------
    fn : Callable
        Bootstrap samples are passed to this function.
    sample : array-like
        Original sample.
    **kwargs
        Keywords are forwarded to :func:`resample`.

    Returns
    -------
    np.array
        Results of `fn` applied to each bootstrap sample.
    """
    return np.asarray([fn(x) for x in resample(sample, **kwargs)])


def confidence_interval(
    fn: _tp.Callable,
    sample: _tp.Iterable,
    *,
    cl: float = 0.95,
    ci_method: str = "bca",
    **kwargs: _Kwargs,
) -> _tp.Tuple[float, float]:
    """
    Calculate bootstrap confidence intervals.

    Parameters
    ----------
    fn : callable
        Function to be bootstrapped.
    sample : array-like
        Original sample.
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
    if not 0 < cl < 1:
        raise ValueError("cl must be between zero and one")

    alpha = 1 - cl
    thetas = bootstrap(fn, sample, **kwargs)

    if ci_method == "percentile":
        return _confidence_interval_percentile(thetas, alpha / 2)

    if ci_method == "bca":
        theta = fn(sample)
        j_thetas = jackknife(fn, sample)
        return _confidence_interval_bca(theta, thetas, j_thetas, alpha / 2)

    raise ValueError(
        f"ci_method must be 'percentile' or 'bca', but '{ci_method}' was supplied"
    )


def bias(fn: _tp.Callable, sample: _tp.Iterable, **kwargs: _Kwargs) -> np.ndarray:
    """
    Calculate bias of the function estimate with the bootstrap.

    Parameters
    ----------
    fn : callable
        Function to be bootstrapped.
    sample : array-like
        Original sample.
    **kwargs
        Keyword arguments forwarded to :func:`resample`.

    Returns
    -------
    ndarray
        Bootstrap estimate of bias (= expectation of estimator - true value).

    Notes
    -----
    This function has special space requirements, it needs to hold `size` replicates of
    the original sample in memory at once. The balanced bootstrap is recommended over
    the ordinary bootstrap for bias estimation, it tends to converge faster.
    """
    replicates = []
    thetas = []
    for b in resample(sample, **kwargs):
        replicates.append(b)
        thetas.append(fn(b))
    population_theta = fn(np.concatenate(replicates))
    return np.mean(thetas, axis=0) - population_theta


def bias_corrected(
    fn: _tp.Callable, sample: _tp.Iterable, **kwargs: _Kwargs
) -> np.ndarray:
    """
    Calculate bias-corrected estimate of the function with the bootstrap.

    Parameters
    ----------
    fn : callable
        Estimator. Can be any mapping ℝⁿ → ℝᵏ, where n is the sample size
        and k is the length of the output array.
    sample : array-like
        Original sample.
    **kwargs
        Keyword arguments forwarded to :func:`resample`.

    Returns
    -------
    ndarray
        Estimate with some bias removed.
    """
    return fn(sample) - bias(fn, sample, **kwargs)


def variance(fn: _tp.Callable, sample: _tp.Iterable, **kwargs: _Kwargs) -> np.ndarray:
    """
    Calculate bootstrap estimate of variance.

    Parameters
    ----------
    fn : callable
        Estimator. Can be any mapping ℝⁿ → ℝᵏ, where n is the sample size
        and k is the length of the output array.
    sample : array-like
        Original sample.
    **kwargs
        Keyword arguments forwarded to :func:`resample`.

    Returns
    -------
    ndarray
        Bootstrap estimate of variance.
    """
    thetas = bootstrap(fn, sample, **kwargs)
    return np.var(thetas, ddof=1, axis=0)


def _resample_stratified(
    sample: np.ndarray,
    size: int,
    method: str,
    strata: np.ndarray,
    rng: np.random.Generator,
) -> _tp.Generator[np.ndarray, None, None]:
    # call resample on sub-samples and merge the replicates
    sub_samples = [sample[strata == x] for x in np.unique(strata)]
    for sub_replicates in zip(
        *[resample(s, size=size, method=method, random_state=rng) for s in sub_samples]
    ):
        yield np.concatenate(sub_replicates, axis=0)


def _resample_ordinary_1(
    sample: np.ndarray, size: int, rng: np.random.Generator
) -> _tp.Generator[np.ndarray, None, None]:
    # i.i.d. sampling from empirical cumulative distribution of sample
    n = len(sample)
    for _ in range(size):
        yield rng.choice(sample, size=n, replace=True)


def _resample_ordinary_n(
    samples: _tp.List[np.ndarray], size: int, rng: np.random.Generator
) -> _tp.Generator[np.ndarray, None, None]:
    n = len(samples[0])
    indices = np.arange(n)
    for _ in range(size):
        m = rng.choice(indices, size=n, replace=True)
        yield (s[m] for s in samples)


def _resample_balanced_1(
    sample: np.ndarray, size: int, rng: np.random.Generator
) -> _tp.Generator[np.ndarray, None, None]:
    # effectively computes a random permutation of `size` concatenated
    # copies of `sample` and returns `size` equal chunks of that
    n = len(sample)
    indices = rng.permutation(n * size)
    for i in range(size):
        m = indices[i * n : (i + 1) * n] % n
        yield sample[m]


def _resample_balanced_n(
    samples: _tp.List[np.ndarray], size: int, rng: np.random.Generator
) -> _tp.Generator[np.ndarray, None, None]:
    n = len(samples[0])
    indices = rng.permutation(n * size)
    for i in range(size):
        m = indices[i * n : (i + 1) * n] % n
        yield (s[m] for s in samples)


def _fit_parametric_family(
    dist: stats.rv_continuous, sample: np.ndarray
) -> _tp.Tuple[float, ...]:
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
) -> _tp.Generator[np.ndarray, None, None]:
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
) -> _tp.Tuple[float, float]:
    quant = quantile_function_gen(thetas)
    return quant(alpha_half), quant(1 - alpha_half)


def _confidence_interval_bca(
    theta: float, thetas: np.ndarray, j_thetas: np.ndarray, alpha_half: float
) -> _tp.Tuple[float, float]:
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
    den = np.sum(j_thetas ** 2)
    acc = num / (6 * den ** 1.5) if den > 0 else 0.0

    z_low = z_naught + norm.ppf(alpha_half)
    z_high = z_naught + norm.ppf(1 - alpha_half)

    p_low = norm.cdf(z_naught + z_low / (1 - acc * z_low))
    p_high = norm.cdf(z_naught + z_high / (1 - acc * z_high))

    quant = quantile_function_gen(thetas)
    return quant(p_low), quant(p_high)
