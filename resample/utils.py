import numpy as np


def ecdf(a):
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


def mise(f, g, d, n=100):
    """
    Estimate mean integrated squared error
    between two functions using Riemann sums.

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
    if d[0] > d[1]:
        raise ValueError("Invalid domain,"
                         " lower bound cannot"
                         " exceed upper bound.")

    p = np.linspace(d[0], d[1], n, endpoint=False)
    w = (d[1] - d[0]) / n

    return np.sum([w * (f(i) - g(i))**2 for i in p])


def sup_norm(f, g, d, n=100):
    """
    Estimate supremum norm of the difference
    of two functions.

    Parameters
    ----------
    f : callable
        First function
    g : callable
        Second function
    d : (float, float)
        Domain
    n : int
        Number of points

    Returns
    -------
    y : float
        Estimated supremum norm
    """
    if d[0] > d[1]:
        raise ValueError("Invalid domain,"
                         " lower bound cannot"
                         " exceed upper bound.")

    p = np.linspace(d[0], d[1], n, endpoint=False)

    return np.max([abs(f(i) - g(i)) for i in p])
