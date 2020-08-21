Changelog
=========

1.0.0 (August xx, 2020)
=======================

API Changes
-----------

- Bootstrap and jackknife generators ``resample.bootstrap.resample`` and ``resample.jackknife.resample`` are now exposed to compute replicates lazily (issue 27).
- Jackknife functions have been split into their own namespace ``resample.jackknife`` (issue 22).
- Random number seeding is now done through using ``numpy`` generators rather than a global random state, which means the minimum ``numpy`` version is now 1.17 (issue 5).
- Parametric bootstrap no longer fixes both parameters of the t distribution (issue 35).
- Default confidence interval method changed from ``"percentile"`` to ``"bca"`` (issue 52).
- Empirical quantile function no longer performs interpolation between quantiles (issue 44).

Enhancements
------------

- Added bootstrap estimate of bias (issue 6).
- Added ``bias_corrected`` function for jackknife and bootstrap, which computes the bias corrected estimates.
- Performance of jackknife computation was increased.

Bug fixes
---------

- Removed incorrect implementation of Studentized bootstrap (issue 49).

Deprecations
------------

- Smoothing of bootstrap samples is no longer supported (issue 7).
- Supremum norm and MISE functionals removed (issue 30).

Other
-----

- Benchmarks were added to track and compare performance of bootstrap and jackknife methods
