from __future__ import division
import numpy as np
from scipy.interpolate import interp1d


def ecdf(a):
    """
    Return the empirical distribution function
    for the given sample

    Parameters
    ----------
    a : array-like
        Sample

    Returns
    -------
    f : callable
        Empirical distribution function
    """
    a = np.sort(a)
    n = len(a)

    def f(x):
        return (np.searchsorted(a, x, side="right",
                                sorter=None) / n)

    return f


def eqf(a):
    """
    Return an empirical quantile function
    for the given sample

    Parameters
    ----------
    a : array-like
        Sample

    Returns
    -------
    f : callable
        Empirical quantile function
    """
    a = np.sort(a)
    n = len(a)

    def inv(x):
        return np.float(interp1d([(i + 1.0) / n
                        for i in range(n)], a)(x))

    def f(x):
        if not (0 <= x <= 1):
            raise ValueError("Argument must be"
                             " between zero and one")
        elif x < 1/n:
            return a[0]
        else:
            return inv(x)

    return f


def mise(f, g, d, n=100):
    """
    Estimate mean integrated squared error
    between two functions using Riemann sums

    Parameters
    ----------
    f : callable
        First function

    g : callable
        Second function

    d : (float, float)
        Domain

    n : int, default : 100
        Number of evaluation points

    Returns
    -------
    y : float
        Estimated MISE
    """
    if d[1] <= d[0]:
        raise ValueError("Invalid domain,"
                         " upper bound must be"
                         " strictly greater"
                         " than lower bound")

    p = np.linspace(d[0], d[1], n, endpoint=False)
    w = (d[1] - d[0]) / n

    return np.sum([w * (f(i) - g(i))**2 for i in p])


def sup_norm(f, g, d, n=100):
    """
    Estimate supremum norm of the difference
    of two functions

    Parameters
    ----------
    f : callable
        First function

    g : callable
        Second function

    d : (float, float)
        Domain

    n : int, default : 100
        Number of evaluation points

    Returns
    -------
    y : float
        Estimated supremum norm
    """
    if d[1] <= d[0]:
        raise ValueError("Invalid domain,"
                         " upper bound must be"
                         " strictly greater"
                         " than lower bound")

    p = np.linspace(d[0], d[1], n, endpoint=False)

    return np.max([abs(f(i) - g(i)) for i in p])
