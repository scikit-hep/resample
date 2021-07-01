.. |resample| image:: doc/_static/logo.svg
   :alt: resample
   :target: http://resample.readthedocs.io

|resample|
==========

.. image:: https://img.shields.io/pypi/v/resample.svg
   :target: https://pypi.org/project/resample

.. image:: https://github.com/resample-project/resample/actions/workflows/run-tests.yml/badge.svg
   :target: https://github.com/resample-project/resample/actions/workflows/run-tests.yml

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
