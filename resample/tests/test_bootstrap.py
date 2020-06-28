import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import pytest
from scipy import stats

from resample.bootstrap import (
    resample,
    bootstrap,
    confidence_interval,
    _fit_parametric_family,
)

PARAMETRIC_CONTINUOUS = {
    # use scipy.stats names here
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
}
PARAMETRIC_DISCRETE = {"poisson"}
PARAMETRIC = PARAMETRIC_CONTINUOUS | PARAMETRIC_DISCRETE
NON_PARAMETRIC = {"ordinary", "balanced"}
ALL_METHODS = NON_PARAMETRIC | PARAMETRIC


@pytest.fixture
def rng():
    return np.random.Generator(np.random.PCG64(1))


@pytest.mark.parametrize("method", ALL_METHODS)
def test_resample_shape_1d(method):
    if method == "beta":
        x = (0.1, 0.2, 0.3)
    else:
        x = (1, 2, 3)
    n_rep = 5
    count = 0
    with np.errstate(invalid="ignore"):
        for bx in resample(x, n_rep, method=method):
            assert len(bx) == len(x)
            count += 1
    assert count == n_rep


@pytest.mark.parametrize("method", NON_PARAMETRIC | {"norm"})
def test_resample_shape_2d(method):
    x = [(1, 2), (4, 3), (6, 5)]
    n_rep = 5
    count = 0
    for bx in resample(x, n_rep, method=method):
        assert bx.shape == np.shape(x)
        count += 1
    assert count == n_rep


@pytest.mark.parametrize("method", NON_PARAMETRIC)
def test_resample_shape_4d(method):
    x = np.ones((2, 3, 4, 5))
    n_rep = 5
    count = 0
    for bx in resample(x, 5, method=method):
        assert bx.shape == np.shape(x)
        count += 1
    assert count == n_rep


@pytest.mark.parametrize("method", NON_PARAMETRIC | PARAMETRIC_CONTINUOUS)
def test_resample_1d_parametric(method, rng):
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

    if method in ("ordinary", "balanced"):
        dist = stats.norm
    else:
        dist = getattr(stats, method)

    x = dist.rvs(*args, size=1000, random_state=rng)

    # make equidistant bins in quantile space for this particular data set
    with np.errstate(invalid="ignore"):
        par = _fit_parametric_family(dist, x)
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
    with np.errstate(invalid="ignore"):
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


def test_resample_1d_parametric_poisson(rng):
    # poisson is behaving super weird in scipy
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


def test_resample_invalid_family_raises():
    msg = "Invalid family"
    with pytest.raises(ValueError, match=msg):
        for _ in resample((1, 2, 3), method="foobar"):
            pass


@pytest.mark.parametrize("method", PARAMETRIC - {"norm"})
def test_resample_2d_parametric_raises(method):
    with pytest.raises(ValueError):
        for _ in resample(np.ones((2, 2)), method=method):
            pass


def test_resample_3d_parametric_normal_raises():
    with pytest.raises(ValueError):
        for _ in resample(np.ones((2, 2, 2)), method="normal"):
            pass


def test_resample_equal_along_axis():
    data = np.reshape(np.tile([0, 1, 2], 3), newshape=(3, 3))
    for b in resample(data, size=2):
        assert_equal(data, b)


@pytest.mark.parametrize("method", NON_PARAMETRIC)
def test_resample_full_strata(method):
    data = np.arange(3)
    for b in resample(data, size=2, strata=data, method=method):
        assert_equal(data, b)


def test_resample_invalid_strata_raises():
    msg = "must have the same shape"
    with pytest.raises(ValueError, match=msg):
        for _ in resample((1, 2, 3), strata=np.arange(4)):
            pass


def test_bootstrap_2d_balanced(rng):
    data = ((1, 2, 3), (2, 3, 4), (3, 4, 5))

    def mean(x):
        return np.mean(x, axis=0)

    r = bootstrap(mean, data, method="balanced")

    # arithmetic mean is linear, therefore mean over all replicas in
    # balanced bootstrap is equal to mean of original sample
    assert_almost_equal(mean(data), mean(r))


@pytest.mark.parametrize("ci_method", ["percentile", "bca", "student"])
def test_confidence_interval(ci_method, rng):
    data = rng.normal(size=1000)
    par = stats.norm.fit(data)
    dist = stats.norm(
        par[0], par[1] / len(data) ** 0.5
    )  # accuracy of mean is sqrt(n) better
    cl = 0.9
    ci_ref = dist.ppf(0.05), dist.ppf(0.95)
    ci = confidence_interval(np.mean, data, cl=cl, size=1000, ci_method=ci_method)
    assert_almost_equal(ci_ref, ci, decimal=2)


def test_confidence_interval_invalid_p_raises():
    msg = "must be between zero and one"
    with pytest.raises(ValueError, match=msg):
        confidence_interval(np.mean, (1, 2, 3), cl=2)


def test_confidence_interval_invalid_ci_method_raises():
    msg = "method must be 'percentile', 'student', or 'bca'"
    with pytest.raises(ValueError, match=msg):
        confidence_interval(np.mean, (1, 2, 3), ci_method="foobar")
