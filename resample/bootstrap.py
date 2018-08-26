import numpy as np


def jackknife(a, f, method="ordinary"):
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
    y : np.array
        Jackknife estimates
    """
    n = len(a)
    X = np.reshape(np.delete(np.tile(a, n),
                             [i * n + i for i in range(n)]),
                   newshape=(n, n - 1))

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
    Calculate estimates from bootstrap samples or
    optionally return bootstrap samples themselves.

    Parameters
    ----------
    a : array-like
        Original sample
    f : callable or None
        Estimator to be bootstrapped, data set
        is return if None
    b : int
        Number of bootstrap samples
    method : string
        * 'ordinary'
        * 'balanced'
        * 'antithetic'

    Returns
    -------
    y | X : np.array
        Estimator applied to each bootstrap sample,
        or bootstrap samples if f is None
    """
    a = np.asarray(a)
    n = len(a)

    if method == "ordinary":
        X = np.reshape(np.random.choice(a, n * b), newshape=(b, n))
    elif method == "balanced":
        X = np.reshape(np.random.permutation(np.repeat(a, b)),
                       newshape=(b, n))
    elif method == "antithetic":
        if f is None:
            raise ValueError("f cannot be None when"
                             " method is 'antithetic'")
        indx = np.argsort(empirical_influence(a, f))
        indx_arr = np.reshape(np.random.choice(indx, size=b // 2 * n),
                              newshape=(b // 2, n))
        n_arr = np.full(shape=(b // 2, n), fill_value=n - 1)
        X = a[np.vstack((indx_arr, n_arr - indx_arr))]
    else:
        raise ValueError("method must be either 'ordinary'"
                         " , 'balanced', or 'antithetic',"
                         " '{method}' was"
                         " supplied".format(method=method))

    if f is None:
        return X
    else:
        return np.apply_along_axis(func1d=f, arr=X, axis=1)

def bootstrap_ci():
    return None
