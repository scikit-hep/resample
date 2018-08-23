import numpy as np


def jackknife(a, func, method="ordinary"):
    """
    Calcualte jackknife estimates for a given sample and
    estimator.

    Parameters
    ----------
    a : array-like
        Sample
    func : callable
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
    y = np.apply_along_axis(func1d=func,
                            arr=X,
                            axis=1)

    return y


def jackknife_bias(a, func, method="ordinary"):
    """
    Jackknife estimate of bias.
    """
    return (len(a) - 1) * np.mean(jackknife(a, func, method=method) - func(a))


def jackknife_variance(a, func, method="ordinary"):
    """
    Jackknife estimate of variance.
    """
    x = jackknife(a, func, method=method)
    return (len(a) - 1) * np.mean((x - np.mean(x))**2)


def empirical_influence(a, func):
    """
    Calculate the empirical influence function for a given
    sample and estimator using the jackknife method.

    Parameters
    ----------
    a : array-like
        Sample
    func : callable
        Estimator

    Returns
    -------
    y : np.array
        Empirical influence values
    """
    theta = func(a)
    theta_j = jackknife(a, func)

    return (len(a) - 1) * (theta - theta_j)


def bootstrap(a, func, b, method="ordinary"):
    """
    Calculate estimates from bootstrap samples.

    Parameters
    ----------
    a : array-like
        Original sample
    func : callable
        Estimator to be bootstrapped
    b : int
        Number of bootstrap samples
    method : string
        * 'ordinary'
        * 'balanced'
        * 'antithetic'

    Returns
    -------
    y : np.array
        Estimator applied to each bootstrap sample
    """
    n = len(a)

    if method == "ordinary":
        X = np.reshape(np.random.choice(a, n * b), newshape=(b, n))
    elif method == "balanced":
        X = np.reshape(np.random.permutation(np.repeat(a, b)),
                       newshape=(b, n))
    elif method == "antithetic":
        indx = np.argsort(empirical_influence(a, func))
        indx_arr = np.reshape(np.random.choice(indx, size=b // 2 * n),
                              newshape=(b // 2, n))
        n_arr = np.full(shape=(b // 2, n), fill_value=n - 1)
        X = a[np.vstack((indx_arr, n_arr - indx_arr))]
    else:
        raise ValueError("method must be either 'ordinary'"
                         " , 'balanced', or 'antithetic',"
                         " '{method}' was"
                         " supplied".format(method=method))

    return np.apply_along_axis(func1d=func, arr=X, axis=1)
