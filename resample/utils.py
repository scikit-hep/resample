"""
Resampling utilities.
"""

from typing import Callable, Tuple
import numpy as np
from scipy.interpolate import interp1d


def empirical_cdf(sample: np.ndarray) -> Callable:
    """
    Return the empirical distribution function for the given sample.

    Parameters
    ----------
    sample : array-like
        Sample.

    Returns
    -------
    callable
        Empirical distribution function.
    """
    sample = np.sort(sample)
    n = len(sample)
    return lambda x: np.searchsorted(sample, x, side="right", sorter=None) / n


def empirical_quantile(sample: np.ndarray) -> Callable:
    """
    Return an empirical quantile function for the given sample.

    Parameters
    ----------
    sample : array-like
        Sample.

    Returns
    -------
    callable
        Empirical quantile function.
    """
    sample = np.sort(sample)
    n = len(sample)
    prob = np.arange(1, n + 1, dtype=float) / n

    def quant(p):
        if not 0 <= p <= 1:
            raise ValueError("Argument must be between zero and one")
        if p < 1 / n:
            # TODO: How to handle this case?
            return sample[0]
        return interp1d(prob, sample)(p)

    return quant


def mise(f: Callable, g: Callable, range: Tuple[float, float], n: int = 100) -> float:
    """
    Estimate mean integrated squared error between two functions using Riemann sums.

    Parameters
    ----------
    f : callable
        First function.

    g : callable
        Second function.

    range : (float, float)
        Domain.

    n : int, default : 100
        Number of evaluation points.

    Returns
    -------
    float
        Estimated MISE.
    """
    if not range[0] < range[1]:
        raise ValueError(
            "Invalid domain, "
            "upper bound must be "
            "strictly greater "
            "than lower bound "
        )

    p = np.linspace(*range, n, endpoint=False)
    w = (range[1] - range[0]) / n

    return np.sum([w * (f(i) - g(i)) ** 2 for i in p])


def sup_norm(
    f: Callable, g: Callable, range: Tuple[float, float], n: int = 100
) -> float:
    """
    Estimate supremum norm of the difference of two functions.

    Parameters
    ----------
    f : callable
        First function.

    g : callable
        Second function.

    range : (float, float)
        Domain.

    n : int, default : 100
        Number of evaluation points.

    Returns
    -------
    float
        Estimated supremum norm.
    """
    if not range[0] < range[1]:
        raise ValueError(
            "Invalid domain, "
            "upper bound must be "
            "strictly greater "
            "than lower bound "
        )

    p = np.linspace(*range, n, endpoint=False)

    return np.max([abs(f(i) - g(i)) for i in p])
