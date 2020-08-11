Changelog
=========

1.0.0 (August xx, 2020)
=======================

Features
--------

- Added bootstrap estimate of bias (:issue:`6`).
- Parametric bootstrap no longer fixes both parameters of the t distribution (:issue:`35`).
- Default confidence interval method changed from ``percentile`` to ``bca`` (:issue:`52`).
- Empirical quantile function no longer performs interpolation between quantiles (:issue:`44`).
- Supremum norm and MISE functionals removed (:issue:`30`).

Bug fixes
---------

- Removed incorrect implementation of Studentized bootstrap (:issue:`49`).

Deprecations
------------

- Smoothing of bootstrap samples is no longer supported (:issue:`7`).
