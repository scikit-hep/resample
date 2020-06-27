"""
Bootstrap resampling.
"""

from typing import Callable, Optional, Tuple, Sequence
import numpy as np
from scipy import stats

from resample.jackknife import jackknife
from resample.utils import empirical_quantile


def resample(
    sample: Sequence,
    size: int,
    method: str = "balanced",
    strata: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generator of bootstrap samples.

    Parameters
    ----------
    sample : array-like
        Original sample.

    size : int, optional
        Number of bootstrap samples to generate. Default is 100.

    method : str or None, optional
        How to generate bootstrap samples. Supported are 'ordinary', 'balanced', or a
        recognized name for a statistical distribution for a parametric bootstrap.
        Default is 'balanced'.

        Supported distribution names: 'gaussian' (alternative: 'norm'), 't'
        (alternative: 'student'), 'laplace', 'logistic', 'F' (alternative: 'f'),
        'gamma', 'log-normal' (alternative: 'lognorm'), 'inverse-gaussian'
        (alternative: 'invgauss'), 'pareto', 'beta', 'poisson'.

    strata : array-like or None
        Stratification labels. Default is None.

    rng : np.random.Generator, optional
        Random number generator instance. Default is the default generator of numpy.

    Yields
    ------
    np.ndarray
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

    if rng is None:
        rng = np.random.default_rng()

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
        j_thetas = jackknife(sample, fcn)
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
    for _ in range(size):
        yield rng.choice(sample, size)


def _resample_balanced(
    sample: np.ndarray,
    size: int,
    strata: Optional[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    if strata is not None:
        raise NotImplementedError

    # effectively computes a random permutation of `size`
    # concatenated copies of `sample`
    n_sample = len(sample)
    indices = rng.permutation(n_sample * size)
    for i in range(size):
        sel = indices[i * n_sample : (i + 1) * n_sample]
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
        "log-normal": stats.lognorm,
        "inverse-gaussian": stats.invgauss,
    }.get(family, getattr(stats, family.lower()))

    if dist is stats.t:
        fit_kwd = {"fscale": 1}
    elif dist is stats.f:
        fit_kwd = {"floc": 0, "fscale": 1}
    elif dist in (stats.gamma, stats.lognorm, stats.invgauss, stats.pareto):
        fit_kwd = {"floc": 0}
    else:
        fit_kwd = {}

    # fit parameters by maximum likelihood and sample from that
    dist = dist(*dist.fit(sample, **fit_kwd))
    for _ in range(size):
        yield dist.rvs(sample.shape, random_state=rng)


def _confidence_interval_percentile(
    thetas: np.ndarray, alpha_half: float,
) -> Tuple[float, float]:
    quantile = empirical_quantile(thetas)
    return quantile(alpha_half), quantile(1 - alpha_half)


def _confidence_interval_studentized(
    theta: float, thetas: np.ndarray, alpha_half: float,
) -> Tuple[float, float]:
    theta_std = np.std(thetas)
    # quantile function of studentized bootstrap estimates
    z = (thetas - theta) / theta_std
    q = empirical_quantile(z)
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

    q = empirical_quantile(thetas)
    return q(p_low), q(p_high)
