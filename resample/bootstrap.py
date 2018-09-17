from __future__ import division
import numpy as np
from scipy.stats import (norm, laplace,
                         gamma, f as F,
                         t, beta, lognorm,
                         pareto, logistic,
                         invgauss, poisson)
from resample.utils import eqf


def jackknife(a, f=None):
    """
    Calculate jackknife estimates for a given sample
    and estimator, return leave-one-out samples
    if estimator is not specified

    Parameters
    ----------
    a : array-like
        Sample

    f : callable or None, default : None
        Estimator

    Returns
    -------
    y | X : np.array
        Jackknife estimates or leave-one-out samples
    """
    arr = np.asarray([a] * len(a))
    X = np.asarray([np.delete(x, i, 0) for i, x in enumerate(arr)])

    if f is None:
        return X
    else:
        return np.asarray([f(x) for x in X])


def jackknife_bias(a, f):
    """
    Calculate jackknife estimate of bias

    Parameters
    ----------
    a : array-like
        Sample

    f : callable
        Estimator

    Returns
    -------
    y : float
        Jackknife estimate of bias
    """
    return (len(a) - 1) * np.mean(jackknife(a, f) - f(a))


def jackknife_variance(a, f):
    """
    Calculate jackknife estimate of variance

    Parameters
    ----------
    a : array-like
        Sample

    f : callable
        Estimator

    Returns
    -------
    y : float
        Jackknife estimate of variance
    """
    x = jackknife(a, f)

    return (len(a) - 1) * np.mean((x - np.mean(x))**2)


def empirical_influence(a, f):
    """
    Calculate the empirical influence function for a given
    sample and estimator using the jackknife method

    Parameters
    ----------
    a : array-like
        Sample

    f : callable
        Estimator

    Returns
    -------
    y : np.array
        Empirical influence values
    """
    return (len(a) - 1) * (f(a) - jackknife(a, f))


def bootstrap(a, f=None, b=100, method="balanced", family=None,
              strata=None, smooth=False, random_state=None):
    """
    Calculate function values from bootstrap samples or
    optionally return bootstrap samples themselves

    Parameters
    ----------
    a : array-like
        Original sample

    f : callable or None, default : None
        Function to be bootstrapped

    b : int, default : 100
        Number of bootstrap samples

    method : string, {'ordinary', 'balanced', 'parametric'},
            default : 'balanced'
       Bootstrapping method

    family : string or None, {'gaussian', 't','laplace',
            'logistic', 'F', 'gamma', 'log-normal',
            'inverse-gaussian', 'pareto', 'beta',
            'poisson'}, default : None
        Distribution family when performing parametric
        bootstrapping

    strata : array-like or None, default : None
        Stratification labels, ignored when method
        is parametric

    smooth : boolean, default : False
        Whether or not to add Gaussian noise to bootstrap
        samples, ignored when method is parametric

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
            raise ValueError("a and strata must have"
                             " the same length")
        # recursively call bootstrap without stratification
        # on the different strata
        masks = [strata == x for x in np.unique(strata)]
        boot_strata = [bootstrap(a=a[m],
                                 f=None,
                                 b=b,
                                 method=method,
                                 strata=None,
                                 random_state=random_state) for m in masks]
        # concatenate resampled strata along first column axis
        X = np.concatenate(boot_strata, axis=1)
    else:
        if method == "ordinary":
            # i.i.d. sampling from ecdf of a
            X = np.reshape(a[np.random.choice(range(a.shape[0]),
                                              a.shape[0] * b)],
                           newshape=(b,) + a.shape)
        elif method == "balanced":
            # permute b concatenated copies of a
            r = np.reshape([a] * b,
                           newshape=(b * a.shape[0],) + a.shape[1:])
            X = np.reshape(r[np.random.permutation(range(r.shape[0]))],
                           newshape=(b,) + a.shape)
        elif method == "parametric":
            if len(a.shape) > 1:
                raise ValueError("a must be one-dimensional")

            # fit parameters by maximum likelihood and sample
            if family == "gaussian":
                theta = norm.fit(a)
                arr = norm.rvs(size=n*b,
                               loc=theta[0],
                               scale=theta[1],
                               random_state=random_state)
            elif family == "t":
                theta = t.fit(a, fscale=1)
                arr = t.rvs(size=n*b,
                            df=theta[0],
                            loc=theta[1],
                            scale=theta[2],
                            random_state=random_state)
            elif family == "laplace":
                theta = laplace.fit(a)
                arr = laplace.rvs(size=n*b,
                                  loc=theta[0],
                                  scale=theta[1],
                                  random_state=random_state)
            elif family == "logistic":
                theta = logistic.fit(a)
                arr = logistic.rvs(size=n*b,
                                   loc=theta[0],
                                   scale=theta[1],
                                   random_state=random_state)
            elif family == "F":
                theta = F.fit(a, floc=0, fscale=1)
                arr = F.rvs(size=n*b,
                            dfn=theta[0],
                            dfd=theta[1],
                            loc=theta[2],
                            scale=theta[3],
                            random_state=random_state)
            elif family == "gamma":
                theta = gamma.fit(a, floc=0)
                arr = gamma.rvs(size=n*b,
                                a=theta[0],
                                loc=theta[1],
                                scale=theta[2],
                                random_state=random_state)
            elif family == "log-normal":
                theta = lognorm.fit(a, floc=0)
                arr = lognorm.rvs(size=n*b,
                                  s=theta[0],
                                  loc=theta[1],
                                  scale=theta[2],
                                  random_state=random_state)
            elif family == "inverse-gaussian":
                theta = invgauss.fit(a, floc=0)
                arr = invgauss.rvs(size=n*b,
                                   mu=theta[0],
                                   loc=theta[1],
                                   scale=theta[2],
                                   random_state=random_state)
            elif family == "pareto":
                theta = pareto.fit(a, floc=0)
                arr = pareto.rvs(size=n*b,
                                 b=theta[0],
                                 loc=theta[1],
                                 scale=theta[2],
                                 random_state=random_state)
            elif family == "beta":
                theta = beta.fit(a)
                arr = beta.rvs(size=n*b,
                               a=theta[0],
                               b=theta[1],
                               loc=theta[2],
                               scale=theta[3],
                               random_state=random_state)
            elif family == "poisson":
                theta = np.mean(a)
                arr = poisson.rvs(size=n*b,
                                  mu=theta,
                                  random_state=random_state)
            else:
                raise ValueError("Invalid family")

            X = np.reshape(arr, newshape=(b, n))
        else:
            raise ValueError("method must be either 'ordinary'"
                             " , 'balanced', or 'parametric',"
                             " '{method}' was supplied".
                             format(method=method))

    # samples are already smooth in the parametric case
    if smooth and (method != "parametric"):
        X += np.random.normal(size=X.shape,
                              scale=1 / np.sqrt(n))

    if f is None:
        return X
    else:
        return np.asarray([f(x) for x in X])


def bootstrap_ci(a, f, p=0.95, b=100, ci_method="percentile",
                 boot_method="balanced", family=None,
                 strata=None, smooth=False, random_state=None):
    """
    Calculate bootstrap confidence intervals

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

    ci_method : string, {'percentile', 'bca', 't'},
            default : 'percentile'
        Confidence interval method

    boot_method : string, {'ordinary', 'balanced',
            'parametric'}, default : 'balanced'
       Bootstrapping method

    family : string or None, {'gaussian', 't','laplace',
            'logistic', 'F', 'gamma', 'log-normal',
            'inverse-gaussian', 'pareto', 'beta',
            'poisson'}, default : None
        Distribution family when performing parametric
        bootstrapping

    strata : array-like or None, default : None
        Stratification labels, ignored when boot_method
        is parametric

    smooth : boolean, default : False
        Whether or not to add Gaussian noise to bootstrap
        samples, ignored when method is parametric

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
        raise ValueError(("boot_method must be 'ordinary'"
                          " 'balanced', or 'parametric', '{method}' was"
                          " supplied".
                         format(method=boot_method)))

    boot_est = bootstrap(a=a, f=f, b=b, method=boot_method,
                         family=family, strata=strata,
                         smooth=smooth, random_state=random_state)

    q = eqf(boot_est)
    alpha = 1 - p

    if ci_method == "percentile":
        return (q(alpha/2), q(1 - alpha/2))
    elif ci_method == "bca":
        theta = f(a)
        # bias correction
        z_naught = norm.ppf(np.mean(boot_est <= theta))
        z_low = norm.ppf(alpha)
        z_high = norm.ppf(1 - alpha)
        # acceleration
        jack_est = jackknife(a, f)
        jack_mean = np.mean(jack_est)
        acc = (np.sum((jack_mean - jack_est)**3) /
               (6 * np.sum((jack_mean - jack_est)**2)**(3/2)))
        p1 = (norm.cdf(z_naught + (z_naught + z_low) /
                       (1 - acc * (z_naught + z_low))))
        p2 = (norm.cdf(z_naught + (z_naught + z_high) /
                       (1 - acc * (z_naught + z_high))))

        return (q(p1), q(p2))
    elif ci_method == "t":
        theta = f(a)
        theta_std = np.std(boot_est)
        # quantile function of studentized bootstrap estimates
        tq = eqf((boot_est - theta) / theta_std)
        t1 = tq(1 - alpha)
        t2 = tq(alpha)

        return (theta - theta_std * t1, theta - theta_std * t2)
    else:
        raise ValueError(("ci_method must be 'percentile'"
                          " 'bca', or 't', '{method}'"
                          " was supplied".
                          format(method=ci_method)))
