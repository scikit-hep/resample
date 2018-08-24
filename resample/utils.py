import numpy as np


def calc_ecdf(a):
    """
    Return the empirical distribution
    function for a given sample.

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

    return (lambda x:
            np.searchsorted(a, x, side="right", sorter=None) / n)


def mise(f, g, mesh):
    """
    Calculate approximate mean integrated squared
    error between functions f and g across a given mesh
    or points

    Parameters
    ----------
    f : callable
        First function
    g : callable
        Second function
    mesh : array-like
        Points at which to evaluate squared error

    Returns
    -------
    y : float
        Mean integrated squared error estimate
    """

    return np.mean([(f(i) - g(i))**2 for i in mesh])
