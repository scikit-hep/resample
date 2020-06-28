"""
Bootstrap resampling.
"""
from typing import Callable, Optional, Tuple, Sequence, Union
import numpy as np
from scipy import stats
from resample.jackknife import jackknife as _jackknife
from resample.empirical import quantile as _quantile


def resample(
    sample: Sequence,
    size: int = 1000,
    method: str = "balanced",
    strata: Optional[np.ndarray] = None,
    random_state: Optional[Union[np.random.Generator, int]] = None,
) -> np.ndarray:
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
    random_state : np.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Yields
    ------
    ndarray
        Bootstrap sample.
    """
    # Stratification:
    # Data is not iid, but consists of several distinct classes. Stratification
    # makes sure that each class is resampled independently, to maintain the
    # relative proportions of each class in the bootstrapped sample.

    sample = np.atleast_1d(sample)
    if strata is not None:
        strata = np.atleast_1d(strata)
        if strata.shape != sample.shape:
            raise ValueError("a and strata must have the same shape")

    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    if method == "balanced":
        return _resample_balanced(sample, size, strata, rng)
    if method == "ordinary":
        return _resample_ordinary(sample, size, strata, rng)
    return _resample_parametric(sample, size, method, strata, rng)


def bootstrap(sample: np.ndarray, fcn: Callable, size: int = 100, **kwds) -> np.ndarray:
    """
    Calculate function values from bootstrap samples.

    Generator of bootstrap samples.

    Parameters
    ----------
    sample : array-like
        Original sample.
    fcn : Callable
        Bootstrap samples are passed to this function.
    **kwds
        Keywords are forwarded to `resample`.

    Returns
    -------
    np.array
        Results of `fcn` applied to each bootstrap sample.
    """
    # if strata is not None and (method != "parametric"):
    #     strata = np.asarray(strata)
    #     if len(strata) != len(a):
    #         raise ValueError("a and strata must have" " the same length")
    #     # recursively call bootstrap without stratification
    #     # on the different strata
    #     masks = [strata == x for x in np.unique(strata)]
    #     boot_strata = [
    #         bootstrap(
    #             a=a[m],
    #             f=None,
    #             b=b,
    #             method=method,
    #             strata=None,
    #             random_state=random_state,
    #         )
    #         for m in masks
    #     ]
    #     # concatenate resampled strata along first column axis
    #     x = np.concatenate(boot_strata, axis=1)
    return np.asarray([fcn(x) for x in resample(sample, size, **kwds)])


def confidence_interval(
    sample: np.ndarray,
    fcn: Callable,
    cl: float = 0.95,
    ci_method: str = "percentile",
    **kwds,
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence intervals.

    Parameters
    ----------
    sample : array-like
        Original sample.
    fcn : callable
        Function to be bootstrapped.
    cl : float, default : 0.95
        Confidence level. Asymptotically, this is the probability that the interval
        contains the true value.
    ci_method : str, {'percentile', 'student', 'bca'}, optional
        Confidence interval method. Default is 'percentile'.
    **kwds
        Keyword arguments forwarded to :func:`bootstrap`.

    Returns
    -------
    (l, u) : tuple
        Upper and lower confidence limits
    """
    if not 0 < cl < 1:
        raise ValueError("cl must be between zero and one")

    alpha = 1 - cl
    thetas = bootstrap(sample, fcn, **kwds)

    if ci_method == "percentile":
        return _confidence_interval_percentile(thetas, alpha)

    if ci_method == "student":
        theta = fcn(sample)
        return _confidence_interval_studentized(theta, thetas, alpha)

    if ci_method == "bca":
        theta = fcn(sample)
        j_thetas = _jackknife(sample, fcn)
        return _confidence_interval_bca(theta, thetas, j_thetas, alpha)

    raise ValueError(
        "ci_method must be 'percentile', 'student', or 'bca', but "
        f"'{ci_method}' was supplied"
    )


def _resample_ordinary(
    sample: np.ndarray,
    size: int,
    strata: Optional[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    if strata is not None:
        raise NotImplementedError

    # i.i.d. sampling from empirical cumulative distribution of sample
    n = len(sample)
    for _ in range(size):
        yield rng.choice(sample, size=n, replace=True)


def _resample_balanced(
    sample: np.ndarray,
    size: int,
    strata: Optional[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    if strata is not None:
        raise NotImplementedError

    # effectively computes a random permutation of `size` concatenated
    # copies of `sample` and returns `size` equal chunks of that
    n = len(sample)
    indices = rng.permutation(n * size)
    for i in range(size):
        sel = indices[i * n : (i + 1) * n] % n
        yield sample[sel]


def _resample_parametric(
    sample: np.ndarray,
    size: int,
    family: str,
    strata: Optional[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    if strata is not None:
        raise NotImplementedError

    dist = {
        # put aliases here
        "gaussian": stats.norm,
        "normal": stats.norm,
        "log-normal": stats.lognorm,
        "log-gaussian": stats.lognorm,
        "inverse-gaussian": stats.invgauss,
        "student": stats.t,
    }.get(family, None)
    if dist is None:
        # use scipy.stats name
        dist = getattr(stats, family.lower())

    if dist == stats.t:
        fit_kwd = {"fscale": 1}  # HD: I think this should not be fixed to 1
    elif dist in {stats.f, stats.beta}:
        fit_kwd = {"floc": 0, "fscale": 1}
    elif dist in (stats.gamma, stats.lognorm, stats.invgauss, stats.pareto):
        fit_kwd = {"floc": 0}
    else:
        fit_kwd = {}

    if sample.ndim > 1:
        if dist is not stats.norm:
            raise ValueError("multivariate data is only supported for 'gaussian'")
        dist = stats.multivariate_normal
        dist.fit = lambda x: (np.mean(x, axis=0), np.cov(x.T, ddof=1))

    n = len(sample)

    # fit parameters by maximum likelihood and sample from that
    if dist == stats.poisson:
        # - poisson has no fit method and there is no scale parameter
        # - random number generation for poisson distribution in scipy seems to be buggy
        mu = np.mean(sample)
        for _ in range(size):
            yield rng.poisson(mu, size=n)
    else:
        args = dist.fit(sample, **fit_kwd)
        dist = dist(*args)
        for _ in range(size):
            yield dist.rvs(size=n, random_state=rng)


def _confidence_interval_percentile(
    thetas: np.ndarray, alpha_half: float,
) -> Tuple[float, float]:
    q = _quantile(thetas)
    return q(alpha_half), q(1 - alpha_half)


def _confidence_interval_studentized(
    theta: float, thetas: np.ndarray, alpha_half: float,
) -> Tuple[float, float]:
    theta_std = np.std(thetas)
    # quantile function of studentized bootstrap estimates
    z = (thetas - theta) / theta_std
    q = _quantile(z)
    theta_std_1 = theta_std * q(1 - alpha_half)
    theta_std_2 = theta_std * q(alpha_half)
    return theta - theta_std_1, theta + theta_std_2


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

    q = _quantile(thetas)
    return q(p_low), q(p_high)
