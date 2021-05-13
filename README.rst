.. |resample| image:: doc/_static/logo.svg
   :alt: resample
   :target: http://resample.readthedocs.io

|resample|
==========

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

Dependencies
------------

Installation requires only `numpy`_ and `scipy`_.

Installation
------------

The latest release can be installed from PyPI::

    pip install resample

or using conda::

    conda install resample -c conda-forge

.. _numpy: http://www.numpy.org
.. _scipy: https://www.scipy.org
