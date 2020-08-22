Changelog
=========

1.0.0 (August xx, 2020)
=======================

API Changes
-----------

- Bootstrap and jackknife generators ``resample.bootstrap.resample`` and ``resample.jackknife.resample`` are now exposed to compute replicates lazily.
- Jackknife functions have been split into their own namespace ``resample.jackknife``.
- Empirical distribution helper functions moved to a ``resample.empirical`` namespace.
- Random number seeding is now done through using ``numpy`` generators rather than a global random state. As a result the minimum ``numpy`` version is now 1.17.
- Parametric bootstrap now estimates both parameters of the t distribution.
- Default confidence interval method changed from ``"percentile"`` to ``"bca"``.
- Empirical quantile function no longer performs interpolation between quantiles.

Enhancements
------------

- Added bootstrap estimate of bias.
- Added ``bias_corrected`` function for jackknife and bootstrap, which computes the bias corrected estimates.
- Performance of jackknife computation was increased.

Bug fixes
---------

- Removed incorrect implementation of Studentized bootstrap.

Deprecations
------------

- Smoothing of bootstrap samples is no longer supported.
- Supremum norm and MISE functionals removed.

Other
-----

- Benchmarks were added to track and compare performance of bootstrap and jackknife methods.
