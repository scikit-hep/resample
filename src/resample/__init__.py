"""
Resampling tools for Python.

This library offers randomisation-based inference in Python based on data resampling
and permutation. The following functionality is implemented.

- Bootstrap samples (ordinary or balanced with optional stratification) from N-D arrays
- Apply parametric bootstrap (Gaussian, Poisson, gamma, etc.) on samples
- Compute bootstrap confidence intervals (percentile or BCa) for any estimator
- Jackknife estimates of bias and variance of any estimator
- Permutation-based variants of traditional statistical tests (t-test, K-S test, etc.)
- Tools for working with empirical distributions (CDF, quantile, etc.)
"""

from importlib.metadata import version

from resample import bootstrap, empirical, jackknife, permutation

__version__ = version("resample")

__all__ = [
    "jackknife",
    "bootstrap",
    "permutation",
    "empirical",
    "__version__",
]
