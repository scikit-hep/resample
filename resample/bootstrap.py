"""
Bootstrap resampling.
"""
from typing import Callable, Optional, Tuple, Sequence, Union, Generator
import numpy as np
from scipy import stats
from resample.jackknife import jackknife as _jackknife
from resample.empirical import quantile_fn as _quantile_fn


def resample(
    sample: Sequence,
    size: int = 1000,
    method: str = "balanced",
    strata: Optional[Sequence] = None,
    random_state: Optional[Union[np.random.Generator, int]] = None,
) -> Generator[np.ndarray, None, None]:
    """
    Generator of bootstrap samples.

    Parameters
    ----------
    sample : array-like
        Original sample.
    size : int, optional
        Number of bootstrap samples to generate. Default is 1000.
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

    Notes
    -----
    Stratification:

    If data is not iid, but consists of several distinct classes, stratification
    ensures that the relative proportions of each class are maintained in each
    replicated sample. This is a stricter constraint than that offered by the
    balanced bootstrap, which only guarantees that classes have the original
    proportions over all replicas.

    Yields
    ------
    ndarray
        Bootstrap sample.
    """
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    sample = np.atleast_1d(sample)

    if strata is not None:
        strata = np.atleast_1d(strata)
        if strata.shape != sample.shape:  # type: ignore
            raise ValueError("a and strata must have the same shape")
        return _resample_stratified(sample, size, method, strata, rng)

    if method == "balanced":
        return _resample_balanced(sample, size, rng)
    if method == "ordinary":
        return _resample_ordinary(sample, size, rng)

    dist = {
        # put aliases here
        "gaussian": stats.norm,
        "normal": stats.norm,
        "log-normal": stats.lognorm,
        "log-gaussian": stats.lognorm,
        "inverse-gaussian": stats.invgauss,
        "student": stats.t,
    }.get(method, None)

    if dist is None:
        # use scipy.stats name
        try:
            dist = getattr(stats, method.lower())
        except AttributeError:
            raise ValueError(f"Invalid family: '{method}'")

    if sample.ndim > 1:
        if dist != stats.norm:
            raise ValueError(f"family '{method}' only supports 1D samples")
        if sample.ndim > 2:
            raise ValueError("multivariate normal only works with 2D samples")
        dist = stats.multivariate_normal

    return _resample_parametric(sample, size, dist, rng)


def bootstrap(fn: Callable, sample: Sequence, size: int = 100, **kwargs) -> np.ndarray:
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
    return np.asarray([fn(x) for x in resample(sample, size, **kwargs)])


def confidence_interval(
    fn: Callable,
    sample: Sequence,
    cl: float = 0.95,
    ci_method: str = "percentile",
    **kwargs,
) -> Tuple[float, float]:
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
    ci_method : str, {'percentile', 'student', 'bca'}, optional
        Confidence interval method. Default is 'percentile'.
    **kwargs
        Keyword arguments forwarded to :func:`resample`.

    Returns
    -------
    (l, u) : tuple
        Upper and lower confidence limits
    """
    if not 0 < cl < 1:
        raise ValueError("cl must be between zero and one")

    alpha = 1 - cl
    thetas = bootstrap(fn, sample, **kwargs)

    if ci_method == "percentile":
        return _confidence_interval_percentile(thetas, alpha / 2)

    theta = fn(sample)

    if ci_method == "student":
        return _confidence_interval_studentized(theta, thetas, alpha / 2)

    if ci_method == "bca":
        j_thetas = _jackknife(fn, sample)
        return _confidence_interval_bca(theta, thetas, j_thetas, alpha / 2)

    raise ValueError(
        "ci_method must be 'percentile', 'student', or 'bca', but "
        f"'{ci_method}' was supplied"
    )


def bias(fn: Callable, sample: Sequence, **kwds) -> np.ndarray:
    """
    Calculate bias of the function estimate with the bootstrap.

    Parameters
    ----------
    fn : callable
        Function to be bootstrapped.
    sample : array-like
        Original sample.
    **kwds
        Keyword arguments forwarded to :func:`resample`.

    Notes
    -----
    The bootstrap method cannot correct biases of linear estimators which are exactly
    of order 1/N, where N is the number of i.i.d. observations, the jackknife should be
    used for those. For bias estimation, the balanced bootstrap is strongly
    recommended. The ordinary bootstrap may requires a much larger number of replicas to
    produce a similarly accurate result.

    Returns
    -------
    ndarray
        Bootstrap estimate of bias (= expectation of estimator - true value).
    """
    theta = fn(sample)
    thetas = bootstrap(fn, sample, **kwds)
    return np.mean(thetas, axis=0) - theta


def bias_corrected(fn: Callable, sample: Sequence, **kwds) -> np.ndarray:
    """
    Calculate bias-corrected estimate of the function with the bootstrap.

    Parameters
    ----------
    fn : callable
        Estimator. Can be any mapping ℝⁿ → ℝᵏ, where n is the sample size
        and k is the length of the output array.
    sample : array-like
        Original sample.
    **kwds
        Keyword arguments forwarded to :func:`resample`.

    Returns
    -------
    ndarray
        Estimate with some bias removed.
    """

    theta = fn(sample)
    thetas = bootstrap(fn, sample, **kwds)
    # bias = mean(thetas) - theta
    # bias-corrected = theta - bias = 2 theta - mean(thetas)
    return 2 * theta - np.mean(thetas, axis=0)


def variance(fn: Callable, sample: Sequence, **kwargs) -> np.ndarray:
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
) -> Generator[np.ndarray, None, None]:
    # call resample on sub-samples and merge the replicas
    sub_samples = [sample[strata == x] for x in np.unique(strata)]
    for sub_replicas in zip(
        *[resample(s, size, method=method, random_state=rng) for s in sub_samples]
    ):
        yield np.concatenate(sub_replicas, axis=0)


def _resample_ordinary(
    sample: np.ndarray, size: int, rng: np.random.Generator,
) -> Generator[np.ndarray, None, None]:
    # i.i.d. sampling from empirical cumulative distribution of sample
    n = len(sample)
    for _ in range(size):
        yield rng.choice(sample, size=n, replace=True)


def _resample_balanced(
    sample: np.ndarray, size: int, rng: np.random.Generator,
) -> Generator[np.ndarray, None, None]:
    # effectively computes a random permutation of `size` concatenated
    # copies of `sample` and returns `size` equal chunks of that
    n = len(sample)
    indices = rng.permutation(n * size)
    for i in range(size):
        sel = indices[i * n : (i + 1) * n] % n
        yield sample[sel]


def _fit_parametric_family(dist: stats.rv_continuous, sample: np.ndarray) -> Tuple:
    if dist == stats.multivariate_normal:
        # has no fit method...
        return np.mean(sample, axis=0), np.cov(sample.T, ddof=1)

    if dist in {stats.f, stats.beta}:
        fit_kwargs = {"floc": 0, "fscale": 1}
    elif dist in {stats.gamma, stats.lognorm, stats.invgauss, stats.pareto}:
        fit_kwargs = {"floc": 0}
    else:
        fit_kwargs = {}

    return dist.fit(sample, **fit_kwargs)


def _resample_parametric(
    sample: np.ndarray, size: int, dist, rng: np.random.Generator,
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
    thetas: np.ndarray, alpha_half: float,
) -> Tuple[float, float]:
    q = _quantile_fn(thetas)
    return q(alpha_half), q(1 - alpha_half)


def _confidence_interval_studentized(
    theta: float, thetas: np.ndarray, alpha_half: float,
) -> Tuple[float, float]:
    theta_std = np.std(thetas)
    # quantile function of studentized bootstrap estimates
    z = (thetas - theta) / theta_std
    q = _quantile_fn(z)
    theta_std_1 = theta_std * q(alpha_half)
    theta_std_2 = theta_std * q(1 - alpha_half)
    return theta + theta_std_1, theta + theta_std_2


def _confidence_interval_bca(
    theta: float, thetas: np.ndarray, j_thetas: np.ndarray, alpha_half: float
) -> Tuple[float, float]:
    norm = stats.norm

    # bias correction
    prop_less = np.mean(thetas < theta)  # proportion of replicas less than obs
    z_naught = norm.ppf(prop_less)

    # acceleration
    j_thetas -= np.mean(j_thetas)
    num = np.sum((-j_thetas) ** 3)
    den = np.sum(j_thetas ** 2)
    acc = num / (6 * den ** 1.5)

    z_low = z_naught + norm.ppf(alpha_half)
    z_high = z_naught + norm.ppf(1 - alpha_half)

    p_low = norm.cdf(z_naught + z_low / (1 - acc * z_low))
    p_high = norm.cdf(z_naught + z_high / (1 - acc * z_high))

    q = _quantile_fn(thetas)
    return q(p_low), q(p_high)
