import numpy as np


def calc_ecdf(a):
    """
    Return the empirical distribution function
    for a given sample.

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


def mise(f, g, cmin, cmax, n):
    """
    Estimate mean integrated squared error
    between two functions using Riemann sums.

    Parameters
    ----------
    f : callable
        First function
    g : callable
        Second function
    cmin : int
        Left endpoint
    cmax : int
        Right endpoint
    n : int
        Number of evaluation points

    Returns
    -------
    y : float
        Estimated MISE
    """
    p = np.linspace(cmin, cmax, n, endpoint=False)
    w = (cmax - cmin) / n

    return np.sum([w * (f(i) - g(i))**2 for i in p])
