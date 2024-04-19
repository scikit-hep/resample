Changelog
=========

For more recent versions, please look into `the release notes on Github <https://github.com/scikit-hep/resample/releases>`_.

1.5.1 (March 1, 2022)
---------------------

- Documentation improvements

1.5.0 (March 1, 2022)
---------------------

This is an API breaking release. The backward-incompatible changes are limited to the
``resample.permutation`` submodule. Other modules are not affected.

Warning: It was discovered that the tests implemented in ``resample.permutation`` had
various issues and gave wrong results, so any results obtained with these tests should
be revised. Since the old implementations were broken anyway, the API of
``resample.permutation`` was altered to fix some design issues as well.

Installing resample now requires compiling a C extension. This is needed for the
computation of the new USP test. This makes the installation of this package less
convenient, since now a C compiler is required on the target machine (or we have to
start building binary wheels). The plan is to move the compiled code to SciPy, which
would allows us to drop the C extension again in the future.

New features
~~~~~~~~~~~~
- ``resample`` functions in ``resample.bootstrap`` and ``resample.jackknife``, and all
  convenience functions which depend on them, can now resample from several arrays of
  equal length, example: ``for ai, bi in resample(a, b): ...``
- USP test of independence for binned data was added to ``resample.permutation``
- ``resample.permutation.same_population`` was added as a generic permutation-based test
  to compute the p-value that two or more samples were drawn from the same population

API changes in submodule ``resample.permutation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- All functions now only accept keyword arguments for their configuration and return a
  tuple-like ``TestResult`` object instead of a dictionary
- Keyword ``b`` in all tests was replaced by ``size``
- p-values computed by all tests are now upper limits to the true Type I error rate
- ``corr_test`` was replaced by two separate functions ``pearsonr`` and ``spearmanr``
- ``kruskal_wallis`` was renamed to ``kruskal`` to follow SciPy naming
- ``anova`` and ``kruskal`` now accept multiple samples directly instead of using a list
  of samples; example: ``anova(x, y)`` instead of ``anova([x, y])``
- ``wilcoxon`` and ``ks_test`` were removed, since the corresponding tests in Scipy
  (``wilcoxon`` and ``ks_2samp``) compute exact p-values for small samples and use
  asymptotic formulas only when the same size is large; we cannot do better than that

1.0.1 (August 23, 2020)
-----------------------

- Minor fix to allow building from source.

1.0.0 (August 22, 2020)
-----------------------

API Changes
~~~~~~~~~~~

- Bootstrap and jackknife generators ``resample.bootstrap.resample`` and ``resample.jackknife.resample`` are now exposed to compute replicates lazily.
- Jackknife functions have been split into their own namespace ``resample.jackknife``.
- Empirical distribution helper functions moved to a ``resample.empirical`` namespace.
- Random number seeding is now done through using ``numpy`` generators rather than a global random state. As a result the minimum ``numpy`` version is now 1.17.
- Parametric bootstrap now estimates both parameters of the t distribution.
- Default confidence interval method changed from ``"percentile"`` to ``"bca"``.
- Empirical quantile function no longer performs interpolation between quantiles.

Enhancements
~~~~~~~~~~~~

- Added bootstrap estimate of bias.
- Added ``bias_corrected`` function for jackknife and bootstrap, which computes the bias corrected estimates.
- Performance of jackknife computation was increased.

Bug fixes
~~~~~~~~~

- Removed incorrect implementation of Studentized bootstrap.

Deprecations
~~~~~~~~~~~~

- Smoothing of bootstrap samples is no longer supported.
- Supremum norm and MISE functionals removed.

Other
~~~~~

- Benchmarks were added to track and compare performance of bootstrap and jackknife methods.
