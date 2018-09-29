Bootstrap
=========

Features
--------

  * Ordinary, balanced and parametric bootstrapping
  * Stratification
  * Smoothing
  * Bootstrap confidence intervals (percentile, BCA, Studentized)
  * Jackknife samples and jackknife estimates of bias and variance
  * Empirical influence functions

Examples
--------

    >>> import numpy as np
    >>> from resample.bootstrap import bootstrap, bootstrap_ci
    >>> np.random.seed(2357)
    >>> x = np.random.randn(100)
    >>> bootstrap(x)
    array([[-0.34091204, -0.86582491,  1.29133372, ...,  0.54245675,
             2.42339546,  0.61766268],
           [ 2.00772156,  0.99849406,  0.03062839, ...,  0.8314122 ,
             0.66923013, -1.3125923 ],
           [-0.29620011,  0.71655732,  0.79604422, ..., -0.45277243,
            -0.24169364,  1.29133372],
           ...,
           [-0.93009922, -1.3125923 ,  0.60603591, ..., -0.29620011,
            -0.34091204,  0.55999393],
           [-0.96238002,  0.67878825,  0.60603591, ...,  2.82713752,
             0.89119024,  0.19951248],
           [-0.499894  ,  0.48898046,  0.88077465, ...,  0.91139591,
            -0.1112024 ,  1.03228398]])
    >>> bootstrap_ci(x, f=np.var)
    (0.6298945281864454, 1.2492297121619806)

Docstrings
----------

.. automodule:: resample.bootstrap
    :members:
