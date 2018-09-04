# resample

Description
--------

`resample` provides a set of tools for performing nonparametric statistical inference in Python, primarily through the use of randomization tests and bootstrapping methods.  See the [example notebook](https://github.com/dsaxton/resample/blob/master/doc/resample.ipynb) for a brief tutorial.

Features
--------

* Bootstrap samples (ordinary or balanced, both with optional stratification) of arrays with arbitrary dimension 
* Bootstrap confidence intervals (percentile, BCA and Studentized) for any well-defined parameter
* Randomization-based variants of traditional statistical tests (t-test, ANOVA F-test, K-S test, etc.)
* Tools for working with empirical distributions (empirical cumulative distribution and quantile functions, distance metrics for comparing distributions)

Dependencies
------------

Installation requires [numpy](http://www.numpy.org/) and [scipy](https://www.scipy.org/).

Installation
------------

The latest release can be installed from PyPI:

    pip install resample

