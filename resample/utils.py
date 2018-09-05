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
    n : int
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
    n : int
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


def distcorr(a1, a2):
    """
    Calculate distance correlation between
    two samples

    Parameters
    ----------
    a1 : array-like
        First sample
    a2 : array-like
        Second sample

    Returns
    -------
    y : float
        Distance correlation
    """
    n = len(a1)
    n2 = len(a1)

    if n != n2:
        raise ValueError("Samples must have equal"
                         " length")

    a1 = np.asarray(a1)
    a2 = np.asarray(a2)

    a = np.zeros(shape=(n, n))
    b = np.zeros(shape=(n, n))

    for i in range(n):
        for j in range(i+1, n):
            a[i, j] = abs(a1[i] - a1[j])
            b[i, j] = abs(a2[i] - a2[j])

    a = a + a.T
    b = b + b.T

    a_bar = np.vstack([np.nanmean(a, axis=0)] * n)
    b_bar = np.vstack([np.nanmean(b, axis=0)] * n)

    A = a - a_bar - a_bar.T + np.full(shape=(n, n), fill_value=a_bar.mean())
    B = b - b_bar - b_bar.T + np.full(shape=(n, n), fill_value=b_bar.mean())

    cov_ab = np.sqrt(np.nansum(A * B)) / n
    std_a = np.sqrt(np.sqrt(np.nansum(A**2)) / n)
    std_b = np.sqrt(np.sqrt(np.nansum(B**2)) / n)

    return cov_ab / std_a / std_b
