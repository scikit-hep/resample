import numpy as np


def jackknife(a, f=None, method="ordinary"):
    """
    Calcualte jackknife estimates for a given sample
    and estimator.

    Parameters
    ----------
    a : array-like
        Sample
    f : callable
        Estimator
    method : string
        * 'ordinary'
        * 'infinitesimal'

    Returns
    -------
    y | X : np.array
        Jackknife estimates
    """
    n = len(a)
    X = np.reshape(np.delete(np.tile(a, n),
                             [i * n + i for i in range(n)]),
                   newshape=(n, n - 1))
    
    if f is None:
        return X
    else:
        return np.apply_along_axis(func1d=f,
                                   arr=X,
                                   axis=1)


def jackknife_bias(a, f, method="ordinary"):
    """
    Calculate jackknife estimate of bias.

    Parameters
    ----------
    a : array-like
        Sample
    f : callable
        Estimator
    method : string
        * 'ordinary'
        * 'infinitesimal'

    Returns
    -------
    y : float
        Jackknife estimate of bias
    """
    return (len(a) - 1) * np.mean(jackknife(a, f, method=method) - f(a))


def jackknife_variance(a, f, method="ordinary"):
    """
    Calculate jackknife estimate of variance.

    Parameters
    ----------
    a : array-like
        Sample
    f : callable
        Estimator
    method : string
        * 'ordinary'
        * 'infinitesimal'

    Returns
    -------
    y : float
        Jackknife estimate of variance
    """
    x = jackknife(a, f, method=method)

    return (len(a) - 1) * np.mean((x - np.mean(x))**2)


def empirical_influence(a, f):
    """
    Calculate the empirical influence function for a given
    sample and estimator using the jackknife method.

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


def bootstrap(a, f=None, b=100, method="ordinary"):
    """
    Calculate function values from bootstrap samples or
    optionally return bootstrap samples themselves.

    Parameters
    ----------
    a : array-like
        Original sample
    f : callable or None
        Function to be bootstrapped
    b : int
        Number of bootstrap samples
    method : string
        * 'ordinary'
        * 'balanced'

    Returns
    -------
    y | X : np.array
        Function applied to each bootstrap sample
        or bootstrap samples if f is None
    """
    a = np.asarray(a)

    if method == "ordinary":
        X = np.reshape(a[np.random.choice(range(a.shape[0]),
                                          a.shape[0] * b)],
                       newshape=(b,) + a.shape)
    elif method == "balanced":
        r = np.reshape([a] * b,
                       newshape=(b * a.shape[0],) + a.shape[1:])
        X = np.reshape(r[np.random.permutation(range(r.shape[0]))],
                       newshape=(b,) + a.shape)
    else:
        raise ValueError("method must be either 'ordinary'"
                         " , or 'balanced', '{method}' was"
                         " supplied".format(method=method))

    if f is None:
        return X
    else:
        return np.asarray([f(x) for x in X])


def bootstrap_ci():
    """
    Calculate bootstrap confidence intervals

    Parameters
    ----------
    a : array-like
        Original sample
    f : callable
        Function to be bootstrapped
    b : int
        Number of bootstrap samples
    method : string
        * 'basic'
        * 'percentile'
        * 'studentized'
        * 'bias-corrected'
        * 'accelerated'

    Returns
    -------
    """
    return None
