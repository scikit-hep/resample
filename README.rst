.. |resample| image:: doc/_static/logo.svg
   :alt: resample
   :target: http://resample.readthedocs.io

|resample|
==========

.. skip-marker-do-not-remove

resample provides a set of tools for performing randomisation-based inference in Python through the use of data resampling and Monte Carlo permutation tests.

Features
--------

- Bootstrap samples (ordinary or balanced, both with optional stratification) of arrays with arbitrary  dimension
- Parametric bootstrap samples (Gaussian, Poisson, gamma, etc.) of one-dimensional arrays
- Bootstrap confidence intervals (percentile or BCa) for any well-defined parameter
- Jackknife estimates of bias and variance
- Randomization-based variants of traditional statistical tests (t-test, ANOVA F-test, K-S test, etc.)
- Tools for working with empirical distributions (cumulative distribution, quantile, and influence functions)

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
