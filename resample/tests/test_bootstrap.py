import numpy as np
from numpy.testing import assert_equal
import pytest
from pytest import approx
from scipy import stats
from collections import Counter

from resample.bootstrap import resample, bootstrap, confidence_interval

PARAMETRIC_CONTINUOUS = (
    "norm",
    "t",
    "laplace",
    "logistic",
    "f",
    "beta",
    "gamma",
    "lognorm",
    "invgauss",
    "pareto",
)
PARAMETRIC_DISCRETE = ("poisson",)
PARAMETRIC = PARAMETRIC_CONTINUOUS + PARAMETRIC_DISCRETE
NON_PARAMETRIC = ("ordinary", "balanced")
ALL_METHODS = NON_PARAMETRIC + PARAMETRIC

# n = 100
# b = 100
# x = np.random.rand(n)
# f = ecdf(x)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_resample_shape_1d(method):
    x = (1, 2, 3)
    n_rep = 5
    count = 0
    for bx in resample(x, n_rep, method=method):
        assert len(bx) == len(x)
        count += 1
    assert count == n_rep


@pytest.mark.parametrize("method", NON_PARAMETRIC + ("normal",))
def test_resample_shape_2d(method):
    x = [(1, 2), (4, 3), (6, 5)]
    n_rep = 5
    count = 0
    for bx in resample(x, n_rep, method=method):
        assert bx.shape == np.shape(x)
        count += 1
    assert count == n_rep


@pytest.mark.parametrize("method", NON_PARAMETRIC + PARAMETRIC_CONTINUOUS)
def test_resample_1d_parametric(method):
    # distribution parameters for parametric families
    args = {
        "t": (2,),
        "f": (25, 20),
        "beta": (2, 1),
        "gamma": (1.5,),
        "lognorm": (1.0,),
        "invgauss": (1,),
        "pareto": (1,),
    }.get(method, ())

    # fit conditions, must be in sync with bootstrap._resample_parametric
    fit_kwd = {
        "t": {"fscale": 1},
        "f": {"floc": 0, "fscale": 1},
        "beta": {"floc": 0, "fscale": 1},
        "gamma": {"floc": 0},
        "lognorm": {"floc": 0},
        "invgauss": {"floc": 0},
        "pareto": {"floc": 0},
    }.get(method, {})

    if method in ("ordinary", "balanced"):
        dist = stats.norm
    else:
        dist = getattr(stats, method)

    rng = np.random.Generator(np.random.PCG64(1))

    x = dist.rvs(*args, size=1000, random_state=rng)

    # get MLE parameters for this sample
    par = dist.fit(x, **fit_kwd)

    # make equidistant bins in quantile space
    prob = np.linspace(0, 1, 11)
    xe = dist(*par).ppf(prob)

    # - in case of parametric bootstrap, wref is exactly uniform
    # - in case of ordinary and balanced, it needs to be computed from original sample
    if method in ("ordinary", "balanced"):
        wref = np.histogram(x, bins=xe)[0]
    else:
        wref = len(x) / (len(xe) - 1)

    # compute P values for replicas compared to original
    prob = []
    wsum = 0
    for bx in resample(x, 100, method=method, random_state=rng):
        w = np.histogram(bx, bins=xe)[0]
        wsum += w
        c = stats.chisquare(w, wref)
        prob.append(c.pvalue)

    if method == "balanced":
        # balanced bootstrap exactly reproduces frequencies in original sample
        assert_equal(wref * 100, wsum)

    # check whether P value distribution is flat
    # - test has chance probability of 1 % to fail randomly
    # - if it fails due to programming error, value is typically < 1e-20
    wp = np.histogram(prob, range=(0, 1))[0]
    c = stats.chisquare(wp)
    assert c.pvalue > 0.01


def test_resample_1d_parametric_poisson():
    # poisson is behaving super weird in scipy

    rng = np.random.Generator(np.random.PCG64(1))

    x = rng.poisson(1.5, size=1000)
    mu = np.mean(x)

    xe = (0, 1, 2, 3, 10)
    # somehow location 1 is needed here...
    wref = np.diff(stats.poisson(mu, 1).cdf(xe)) * len(x)

    # compute P values for replicas compared to original
    prob = []
    for bx in resample(x, 100, method="poisson", random_state=rng):
        w = np.histogram(bx, bins=xe)[0]
        c = stats.chisquare(w, wref)
        prob.append(c.pvalue)

    # check whether P value distribution is flat
    # - test has chance probability of 1 % to fail randomly
    # - if it fails due to programming error, value is typically < 1e-20
    wp = np.histogram(prob, range=(0, 1))[0]
    c = stats.chisquare(wp)
    assert c.pvalue > 0.01


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
