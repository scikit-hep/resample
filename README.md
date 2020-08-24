# resample

Description
--------

`resample` provides a set of tools for performing randomization-based inference in Python, primarily through the use of bootstrapping methods and Monte Carlo permutation tests.  Documentation can be found on [Read the Docs](https://resample.readthedocs.io/en/latest/).

Features
--------

* Bootstrap samples (ordinary or balanced, both with optional stratification) of arrays with arbitrary dimension
* Parametric bootstrap samples (Gaussian, Poisson, gamma, etc.) of one-dimensional arrays
* Bootstrap confidence intervals (percentile or BCa) for any well-defined parameter
* Jackknife estimates of bias and variance
* Randomization-based variants of traditional statistical tests (t-test, ANOVA F-test, K-S test, etc.)
* Tools for working with empirical distributions (cumulative distribution, quantile, and influence functions)

Dependencies
------------

Installation requires [numpy](http://www.numpy.org/) and [scipy](https://www.scipy.org/).

Installation
------------

The latest release can be installed from PyPI:

    pip install resample
