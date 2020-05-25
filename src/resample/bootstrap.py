from typing import Callable, Optional, Tuple

import numpy as np
import numba as nb
from scipy.stats import (
    norm,
    laplace,
    gamma,
    f as fdist,
    t,
    beta,
    lognorm,
    pareto,
    logistic,
    invgauss,
    poisson,
)

from resample.utils import eqf


@nb.njit
def _jackknife_generator(a: np.ndarray) -> np.ndarray:
    n = len(a)
    x = np.empty(n - 1, dtype=a.dtype)
    for i in range(n - 1):
        x[i] = a[1 + i]
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
            x[j] = a[j]
        yield x.view(x.dtype)  # must return view to avoid a numba life-time bug


def jackknife(a: np.ndarray, f: Callable) -> np.ndarray:
    """
    Calculate jackknife estimates for a given sample and estimator.

    The jackknife is a linear approximation to the bootstrap. In contrast to the
    bootstrap it is deterministic and does not use random numbers. The caveat is the
    computational cost of the jackknife, which is O(n^2) for n observations, compared
    to O(n x k) for k bootstrap replicas. For large samples, the bootstrap is more
    efficient.

    Parameters
    ----------
    a : array-like
        Sample. Must be one-dimensional.

    f : callable
        Estimator. Can be any mapping ℝⁿ → ℝᵏ, where n is the sample size
        and k is the length of the output array.

    Returns
    -------
    np.ndarray
        Jackknife samples.
    """
    a = np.atleast_1d(a)
    return np.asarray([f(x) for x in _jackknife_generator(a)])


def jackknife_bias(a: np.ndarray, f: Callable) -> np.ndarray:
    """
    Calculate jackknife estimate of bias.

    The bias estimate is accurate to O(n^{-1}), where n is the sample size.
    If the bias is exactly O(n^{-1}), then the estimate is exact.

    Wikipedia:
    https://en.wikipedia.org/wiki/Jackknife_resampling

    Parameters
    ----------
    a : array-like
        Sample. Must be one-dimensional.

    f : callable
        Estimator. Can be any mapping ℝⁿ → ℝᵏ, where n is the sample size
        and k is the length of the output array.

    Returns
    -------
    np.ndarray
        Jackknife estimate of bias.
    """
    mj = np.mean(jackknife(a, f), axis=0)
    return (len(a) - 1) * (mj - f(a))


def jackknife_bias_corrected(a: np.ndarray, f: Callable) -> np.ndarray:
    """
    Calculates bias-corrected estimate of the function with the jackknife.

    Removes a bias of O(n^{-1}), where n is the sample size, leaving bias
    of order O(n^{-2}). If the original function has a bias of exactly
    O(n^{-1})), the corrected result is now unbiased.

    Wikipedia:
    https://en.wikipedia.org/wiki/Jackknife_resampling

    Parameters
    ----------
    a : array-like
        Sample. Must be one-dimensional.

    f : callable
        Estimator. Can be any mapping ℝⁿ → ℝᵏ, where n is the sample size
        and k is the length of the output array.

    Returns
    -------
    np.ndarray
        Estimate with O(n^{-1}) bias removed.
    """
    mj = np.mean(jackknife(a, f), axis=0)
    n = len(a)
    return n * f(a) - (n - 1) * mj


def jackknife_variance(a: np.ndarray, f: Callable) -> np.ndarray:
    """
    Calculate jackknife estimate of variance.

    Wikipedia:
    https://en.wikipedia.org/wiki/Jackknife_resampling

    Parameters
    ----------
    a : array-like
        Sample. Must be one-dimensional.

    f : callable
        Estimator. Can be any mapping ℝⁿ → ℝᵏ, where n is the sample size
        and k is the length of the output array.

    Returns
    -------
    np.ndarray
        Jackknife estimate of variance.
    """
    # formula is (n - 1) / n * sum((fj - mean(fj)) ** 2)
    #   = np.var(fj, ddof=0) * (n - 1)
    fj = jackknife(a, f)
    return (len(a) - 1) * np.var(fj, ddof=0, axis=0)


def empirical_influence(a: np.ndarray, f: Callable) -> np.ndarray:
    """
    Calculate the empirical influence function for a given sample and estimator
    using the jackknife method.

    Parameters
    ----------
    a : array-like
        Sample. Must be one-dimensional.

    f : callable
        Estimator. Can be any mapping ℝⁿ → ℝᵏ, where n is the sample size
        and k is the length of the output array.

    Returns
    -------
    np.ndarray
        Empirical influence values.
    """
    return (len(a) - 1) * (f(a) - jackknife(a, f))


def bootstrap(
    a: np.ndarray,
    f: Optional[Callable] = None,
    b: int = 100,
    method: str = "balanced",
    family: Optional[str] = None,
    strata: Optional[np.ndarray] = None,
    random_state=None,
):
    """
    Calculate function values from bootstrap samples or optionally return
    bootstrap samples themselves.

    Parameters
    ----------
    a : array-like
        Original sample

    f : callable or None, default : None
        Function to be bootstrapped

    b : int, default : 100
        Number of bootstrap samples

    method : str, {'ordinary', 'balanced', 'parametric'},
            default : 'balanced'
       Bootstrapping method

    family : str or None, {'gaussian', 't','laplace',
            'logistic', 'F', 'gamma', 'log-normal',
            'inverse-gaussian', 'pareto', 'beta',
            'poisson'}, default : None
        Distribution family when performing parametric
        bootstrapping

    strata : array-like or None, default : None
        Stratification labels, ignored when method
        is parametric

    random_state : int or None, default : None
        Random number seed

    Returns
    -------
    y | X : np.array
        Function applied to each bootstrap sample
        or bootstrap samples if f is None
    """
    np.random.seed(random_state)
    a = np.asarray(a)
    n = len(a)

    # stratification not meaningful for parametric sampling
    if strata is not None and (method != "parametric"):
        strata = np.asarray(strata)
        if len(strata) != len(a):
            raise ValueError("a and strata must have" " the same length")
        # recursively call bootstrap without stratification
        # on the different strata
        masks = [strata == x for x in np.unique(strata)]
        boot_strata = [
            bootstrap(
                a=a[m],
                f=None,
                b=b,
                method=method,
                strata=None,
                random_state=random_state,
            )
            for m in masks
        ]
        # concatenate resampled strata along first column axis
        x = np.concatenate(boot_strata, axis=1)
    else:
        if method == "ordinary":
            # i.i.d. sampling from ecdf of a
            x = np.reshape(
                a[np.random.choice(range(a.shape[0]), a.shape[0] * b)],
                newshape=(b,) + a.shape,
            )
        elif method == "balanced":
            # permute b concatenated copies of a
            r = np.reshape([a] * b, newshape=(b * a.shape[0],) + a.shape[1:])
            x = np.reshape(
                r[np.random.permutation(range(r.shape[0]))], newshape=(b,) + a.shape
            )
        elif method == "parametric":
            if len(a.shape) > 1:
                raise ValueError("a must be one-dimensional")

            # fit parameters by maximum likelihood and sample
            if family == "gaussian":
                theta = norm.fit(a)
                arr = norm.rvs(
                    size=n * b, loc=theta[0], scale=theta[1], random_state=random_state
                )
            elif family == "t":
                theta = t.fit(a, fscale=1)
                arr = t.rvs(
                    size=n * b,
                    df=theta[0],
                    loc=theta[1],
                    scale=theta[2],
                    random_state=random_state,
                )
            elif family == "laplace":
                theta = laplace.fit(a)
                arr = laplace.rvs(
                    size=n * b, loc=theta[0], scale=theta[1], random_state=random_state
                )
            elif family == "logistic":
                theta = logistic.fit(a)
                arr = logistic.rvs(
                    size=n * b, loc=theta[0], scale=theta[1], random_state=random_state
                )
            elif family == "F":
                theta = fdist.fit(a, floc=0, fscale=1)
                arr = fdist.rvs(
                    size=n * b,
                    dfn=theta[0],
                    dfd=theta[1],
                    loc=theta[2],
                    scale=theta[3],
                    random_state=random_state,
                )
            elif family == "gamma":
                theta = gamma.fit(a, floc=0)
                arr = gamma.rvs(
                    size=n * b,
                    a=theta[0],
                    loc=theta[1],
                    scale=theta[2],
                    random_state=random_state,
                )
            elif family == "log-normal":
                theta = lognorm.fit(a, floc=0)
                arr = lognorm.rvs(
                    size=n * b,
                    s=theta[0],
                    loc=theta[1],
                    scale=theta[2],
                    random_state=random_state,
                )
            elif family == "inverse-gaussian":
                theta = invgauss.fit(a, floc=0)
                arr = invgauss.rvs(
                    size=n * b,
                    mu=theta[0],
                    loc=theta[1],
                    scale=theta[2],
                    random_state=random_state,
                )
            elif family == "pareto":
                theta = pareto.fit(a, floc=0)
                arr = pareto.rvs(
                    size=n * b,
                    b=theta[0],
                    loc=theta[1],
                    scale=theta[2],
                    random_state=random_state,
                )
            elif family == "beta":
                theta = beta.fit(a)
                arr = beta.rvs(
                    size=n * b,
                    a=theta[0],
                    b=theta[1],
                    loc=theta[2],
                    scale=theta[3],
                    random_state=random_state,
                )
            elif family == "poisson":
                theta = np.mean(a)
                arr = poisson.rvs(size=n * b, mu=theta, random_state=random_state)
            else:
                raise ValueError("Invalid family")

            x = np.reshape(arr, newshape=(b, n))
        else:
            raise ValueError(
                "method must be either 'ordinary', "
                "'balanced', or 'parametric', "
                f"'{method}' was supplied"
            )

    if f is None:
        return x
    else:
        return np.asarray([f(s) for s in x])


def bootstrap_ci(
    a: np.ndarray,
    f: Callable,
    p: float = 0.95,
    b: int = 100,
    ci_method: str = "percentile",
    boot_method: str = "balanced",
    family: Optional[str] = None,
    strata: Optional[np.ndarray] = None,
    random_state=None,
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence intervals.

    Parameters
    ----------
    a : array-like
        Original sample

    f : callable
        Function to be bootstrapped

    p : float, default : 0.95
        Confidence level

    b : int, default : 100
        Number of bootstrap samples

    ci_method : str, {'percentile', 'bca', 't'},
            default : 'percentile'
        Confidence interval method

    boot_method : str, {'ordinary', 'balanced',
            'parametric'}, default : 'balanced'
       Bootstrapping method

    family : str or None, {'gaussian', 't','laplace',
            'logistic', 'F', 'gamma', 'log-normal',
            'inverse-gaussian', 'pareto', 'beta',
            'poisson'}, default : None
        Distribution family when performing parametric
        bootstrapping

    strata : array-like or None, default : None
        Stratification labels, ignored when boot_method
        is parametric

    random_state : int or None, default : None
        Random number seed

    Returns
    -------
    (l, u) : tuple
        Upper and lower confidence limits
    """
    if not (0 < p < 1):
        raise ValueError("p must be between zero and one")

    if boot_method not in ["ordinary", "balanced", "parametric"]:
        raise ValueError(
            (
                "boot_method must be 'ordinary', "
                f"'balanced', or 'parametric', '{boot_method}' was "
                "supplied"
            )
        )

    boot_est = bootstrap(
        a=a,
        f=f,
        b=b,
        method=boot_method,
        family=family,
        strata=strata,
        random_state=random_state,
    )

    q = eqf(boot_est)
    alpha = 1 - p

    if ci_method == "percentile":
        return q(alpha / 2), q(1 - alpha / 2)
    elif ci_method == "bca":
        theta = f(a)
        # bias correction
        z_naught = norm.ppf(np.mean(boot_est <= theta))
        z_low = norm.ppf(alpha)
        z_high = norm.ppf(1 - alpha)
        # acceleration
        jack_est = jackknife(a, f)
        jack_mean = np.mean(jack_est)
        acc = np.sum((jack_mean - jack_est) ** 3) / (
            6 * np.sum((jack_mean - jack_est) ** 2) ** (3 / 2)
        )
        p1 = norm.cdf(z_naught + (z_naught + z_low) / (1 - acc * (z_naught + z_low)))
        p2 = norm.cdf(z_naught + (z_naught + z_high) / (1 - acc * (z_naught + z_high)))

        return q(p1), q(p2)
    elif ci_method == "t":
        theta = f(a)
        theta_std = np.std(boot_est)
        # quantile function of studentized bootstrap estimates
        tq = eqf((boot_est - theta) / theta_std)
        t1 = tq(1 - alpha)
        t2 = tq(alpha)

        return theta - theta_std * t1, theta - theta_std * t2
    else:
        raise ValueError(
            (
                "ci_method must be 'percentile', "
                f"'bca', or 't', '{ci_method}' "
                "was supplied"
            )
        )
