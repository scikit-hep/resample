.. |resample| image:: doc/_static/logo.svg
   :alt: resample
   :target: http://resample.readthedocs.io

|resample|
==========

.. image:: https://img.shields.io/pypi/v/resample.svg
   :target: https://pypi.org/project/resample

.. image:: https://github.com/resample-project/resample/actions/workflows/test.yml/badge.svg
   :target: https://github.com/resample-project/resample/actions/workflows/tests.yml

.. image:: https://readthedocs.org/projects/resample/badge/?version=stable
   :target: https://resample.readthedocs.io/en/stable

.. image:: https://img.shields.io/badge/coverage-96%25-green

.. skip-marker-do-not-remove

Randomisation-based inference in Python based on data resampling and permutation.

Features
--------

- Bootstrap samples (ordinary or balanced with optional stratification) from N-D arrays
- Apply parametric bootstrap (Gaussian, Poisson, gamma, etc.) on samples
- Compute bootstrap confidence intervals (percentile or BCa) for any estimator
- Jackknife estimates of bias and variance of any estimator
- Permutation-based variants of traditional statistical tests (t-test, K-S test, etc.)
- Tools for working with empirical distributions (CDF, quantile, etc.)
- Depends only on `numpy`_ and `scipy`_

Example
-------
```py
# bootstrap uncertainty of arithmetic mean
from resample.bootstrap import variance
import numpy as np

d = [1, 2, 6, 3, 5]

print(f"bootstrap {variance(np.mean, d) ** 0.5:.2f} exact {(np.var(d) / len(d)) ** 0.5:.2f}")
# returns: bootstrap 0.82 exact 0.83
```

.. _numpy: http://www.numpy.org
.. _scipy: https://www.scipy.org
