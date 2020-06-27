import numpy as np
import pytest
from scipy import stats

from resample.bootstrap import resample, bootstrap, confidence_interval
from resample.utils import empirical_cdf as ecdf, sup_norm

n = 100
b = 100
x = np.random.rand(n)
f = ecdf(x)


def test_resample_1d_ordinary():
    n = 100
    x = np.arange(n)
    rng = np.random.Generator(np.random.PCG64(1))
    w = 0.0
    n_replicas = 100
    for bx in resample(x, n_replicas, method="ordinary", rng=rng):
        w += np.histogram(bx, bins=n, range=(0, n))[0]
    print(w)
    assert stats.chisquare(w / (n * n_replicas)) == 0


# def test_ordinary_bootstrap_1d():
#     boot = bootstrap(x, size, method="ordinary")
#     assert boot.shape == (b, n)
#
#
# def test_balanced_bootstrap_distributions_equal():
#     xbal = np.ravel(bootstrap(x, method="balanced"))
#     g = ecdf(xbal)
#     assert sup_norm(f, g, (-10, 10)) == 0.0
#
#
# def test_parametric_bootstrap_multivariate_raises():
#     msg = "must be one-dimensional"
#     with pytest.raises(ValueError, match=msg):
#         bootstrap(
#             np.random.normal(size=(10, 2)), method="parametric", family="gaussian"
#         )
#
#
# def test_parametric_bootstrap_invalid_family_raises():
#     msg = "Invalid family"
#     with pytest.raises(ValueError, match=msg):
#         bootstrap(x, method="parametric", family="____")
#
#
# @pytest.mark.parametrize(
#     "family",
#     [
#         "gaussian",
#         "t",
#         "laplace",
#         "logistic",
#         "F",
#         "gamma",
#         "log-normal",
#         "inverse-gaussian",
#         "pareto",
#         "beta",
#         "poisson",
#     ],
# )
# def test_parametric_bootstrap_shape(family):
#     boot = bootstrap(x, b=b, method="parametric", family=family)
#     assert boot.shape == (b, n)
#
#
# def test_bootstrap_shape():
#     arr = np.random.normal(size=(16, 8, 4, 2))
#     boot = bootstrap(arr, b=b)
#     assert boot.shape == (b, 16, 8, 4, 2)
#
#
# def test_bootstrap_equal_along_axis():
#     arr = np.reshape(np.tile([0, 1, 2], 3), newshape=(3, 3))
#     boot = bootstrap(arr, b=10)
#     assert np.all([np.array_equal(arr, a) for a in boot])
#
#
# def test_bootstrap_full_strata():
#     boot = bootstrap(x, b=b, strata=np.array(range(n)))
#     assert np.all([np.array_equal(x, a) for a in boot])
#
#
# def test_bootstrap_invalid_strata_raises():
#     msg = "must have the same length"
#     with pytest.raises(ValueError, match=msg):
#         bootstrap(x, strata=np.arange(len(x) + 1))
#
#
# def test_bootstrap_invalid_method_raises():
#     msg = "method must be either 'ordinary', 'balanced', or 'parametric'"
#     with pytest.raises(ValueError, match=msg):
#         bootstrap(x, method="____")
#
#
# def test_confidence_interval_invalid_p_raises():
#     msg = "p must be between zero and one"
#     with pytest.raises(ValueError, match=msg):
#         confidence_interval(x, np.mean, cl=2)
#
#
# @pytest.mark.parametrize("ci_method", ["percentile", "bca", "t"])
# def test_confidence_interval_len(ci_method):
#     ci = confidence_interval(x, np.mean, ci_method=ci_method)
#     assert len(ci) == 2
#
#
# def test_confidence_interval_invalid_boot_method_raises():
#     msg = "must be 'ordinary', 'balanced', or 'parametric'"
#     with pytest.raises(ValueError, match=msg):
#         confidence_interval(x, np.mean, boot_method="____")
#
#
# def test_confidence_interval_invalid_ci_method_raises():
#     msg = "method must be 'percentile', 'bca', or 't'"
#     with pytest.raises(ValueError, match=msg):
#         confidence_interval(x, np.mean, ci_method="____")
