.. |resample| image:: doc/_static/logo.svg
   :alt: resample
   :target: http://resample.readthedocs.io

|resample|
==========

.. image:: https://img.shields.io/pypi/v/resample.svg
   :target: https://pypi.org/project/resample
.. image:: https://img.shields.io/conda/vn/conda-forge/resample.svg
   :target: https://github.com/conda-forge/resample-feedstock
.. image:: https://github.com/resample-project/resample/actions/workflows/test.yml/badge.svg
   :target: https://github.com/resample-project/resample/actions/workflows/tests.yml
.. image:: https://coveralls.io/repos/github/resample-project/resample/badge.svg
   :target: https://coveralls.io/github/resample-project/resample
.. image:: https://readthedocs.org/projects/resample/badge/?version=stable
   :target: https://resample.readthedocs.io/en/stable
.. image:: https://img.shields.io/pypi/l/resample
   :target: https://pypi.org/project/resample
.. image:: https://zenodo.org/badge/145776396.svg
   :target: https://zenodo.org/badge/latestdoi/145776396

`Link to full documentation`_

.. _Link to full documentation: http://resample.readthedocs.io

.. skip-marker-do-not-remove

Resampling-based inference in Python based on data resampling and permutation.

This package was created by Daniel Saxton and is now maintained by Hans Dembinski.

Features
--------

- Bootstrap resampling: ordinary or balanced with optional stratification
- Extended bootstrap resampling: also varies sample size
- Parametric resampling: Gaussian, Poisson, gamma, etc.)
- Jackknife estimates of bias and variance of any estimator
- Compute bootstrap confidence intervals (percentile or BCa) for any estimator
- Permutation-based variants of traditional statistical tests (**USP test of independence** and others)
- Tools for working with empirical distributions (CDF, quantile, etc.)
- Depends only on `numpy`_ and `scipy`_

Example
-------

We bootstrap the uncertainty of the arithmetic mean, an estimator for the expectation. In this case, we know the formula to compute this uncertainty and can compare it to the bootstrap result. More complex examples can be found `in the documentation <https://resample.readthedocs.io/en/stable/tutorials.html>`_.

.. code-block:: python

      from resample.bootstrap import variance
      import numpy as np

      # data
      d = [1, 2, 6, 3, 5]

      # this call is all you need
      stdev_of_mean = variance(np.mean, d) ** 0.5
      
      print(f"bootstrap {stdev_of_mean:.2f}")
      print(f"exact {np.std(d) / len(d) ** 0.5:.2f}")
      # bootstrap 0.82
      # exact 0.83

The amazing thing is that the bootstrap works as well for arbitrarily complex estimators.
The bootstrap often provides good results even when the sample size is small.

.. _numpy: http://www.numpy.org
.. _scipy: https://www.scipy.org

Installation
------------
You can install with pip.

.. code-block:: shell

      pip install resample
