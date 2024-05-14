# ruff: noqa: D100 D103
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_allclose
from scipy import stats

from resample.bootstrap import (
    _fit_parametric_family,
    bootstrap,
    confidence_interval,
    resample,
    variance,
    covariance,
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


def chisquare(
    obs, exp=None
):  # we do not use scipy.stats.chisquare, because it is broken
    n = len(obs)
    if exp is None:
        exp = 1.0 / n
    t = np.sum(obs**2 / exp) - n
    return stats.chi2(n - 1).cdf(t)


@pytest.fixture
def rng():
    return np.random.Generator(np.random.PCG64(1))


@pytest.mark.parametrize("method", ALL_METHODS)
def test_resample_shape_1d(method):
    if method == "beta":
        x = (0.1, 0.2, 0.3)
    else:
        x = (1.0, 2.0, 3.0)
    n_rep = 5
    count = 0
    with np.errstate(invalid="ignore"):
        for bx in resample(x, size=n_rep, method=method):
            assert len(bx) == len(x)
            count += 1
    assert count == n_rep


@pytest.mark.parametrize("method", NON_PARAMETRIC | {"norm"})
def test_resample_shape_2d(method):
    x = [(1.0, 2.0), (4.0, 3.0), (6.0, 5.0)]
    n_rep = 5
    count = 0
    for bx in resample(x, size=n_rep, method=method):
        assert bx.shape == np.shape(x)
        count += 1
    assert count == n_rep


@pytest.mark.parametrize("method", NON_PARAMETRIC)
def test_resample_shape_4d(method):
    x = np.ones((2, 3, 4, 5))
    n_rep = 5
    count = 0
    for bx in resample(x, size=n_rep, method=method):
        assert bx.shape == np.shape(x)
        count += 1
    assert count == n_rep


@pytest.mark.parametrize("method", NON_PARAMETRIC | PARAMETRIC_CONTINUOUS)
def test_resample_1d_statistical_test(method, rng):
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

    if method in NON_PARAMETRIC:
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
    if method in NON_PARAMETRIC:
        wref = np.histogram(x, bins=xe)[0]
    else:
        wref = len(x) / (len(xe) - 1)

    # compute P values for replicates compared to original
    prob = []
    wsum = 0
    with np.errstate(invalid="ignore"):
        for bx in resample(x, size=100, method=method, random_state=rng):
            w = np.histogram(bx, bins=xe)[0]
            wsum += w
            pvalue = chisquare(w, wref)
            prob.append(pvalue)

    if method == "balanced":
        # balanced bootstrap exactly reproduces frequencies in original sample
        assert_equal(wref * 100, wsum)

    # check whether P value distribution is flat
    # - test has chance probability of 1 % to fail randomly
    # - if it fails due to programming error, value is typically < 1e-20
    wp = np.histogram(prob, range=(0, 1))[0]
    pvalue = chisquare(wp)
    assert pvalue > 0.01


def test_resample_1d_statistical_test_poisson(rng):
    # poisson is behaving super weird in scipy
    x = rng.poisson(1.5, size=1000)
    mu = np.mean(x)

    xe = (0, 1, 2, 3, 10)
    # somehow location 1 is needed here...
    wref = np.diff(stats.poisson(mu, 1).cdf(xe)) * len(x)

    # compute P values for replicates compared to original
    prob = []
    for bx in resample(x, size=100, method="poisson", random_state=rng):
        w = np.histogram(bx, bins=xe)[0]

        pvalue = chisquare(w, wref)
        prob.append(pvalue)

    # check whether P value distribution is flat
    # - test has chance probability of 1 % to fail randomly
    # - if it fails due to programming error, value is typically < 1e-20
    wp = np.histogram(prob, range=(0, 1))[0]
    pvalue = chisquare(wp)
    assert pvalue > 0.01


def test_resample_invalid_family_raises():
    msg = "Invalid family"
    with pytest.raises(ValueError, match=msg):
        next(resample((1, 2, 3), method="foobar"))


@pytest.mark.parametrize("method", PARAMETRIC - {"norm"})
def test_resample_2d_parametric_raises(method):
    with pytest.raises(ValueError):
        next(resample(np.ones((2, 2)), method=method))


def test_resample_3d_parametric_normal_raises():
    with pytest.raises(ValueError):
        next(resample(np.ones((2, 2, 2)), method="normal"))


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
        next(resample((1, 2, 3), strata=np.arange(4)))


def test_bootstrap_2d_balanced(rng):
    data = ((1, 2, 3), (2, 3, 4), (3, 4, 5))

    def mean(x):
        return np.mean(x, axis=0)

    r = bootstrap(mean, data, method="balanced")

    # arithmetic mean is linear, therefore mean over all replicates in
    # balanced bootstrap is equal to mean of original sample
    assert_allclose(mean(data), mean(r))


@pytest.mark.parametrize("action", [bootstrap, variance, confidence_interval])
def test_bootstrap_several_args(action):
    x = [1, 2, 3]
    y = [4, 5, 6]
    xy = np.transpose([x, y])

    if action is confidence_interval:

        def f1(x, y):
            return np.sum(x + y)

        def f2(xy):
            return np.sum(xy)

    else:

        def f1(x, y):
            return np.sum(x), np.sum(y)

        def f2(xy):
            return np.sum(xy, axis=0)

    r1 = action(f1, x, y, size=10, random_state=1)
    r2 = action(f2, xy, size=10, random_state=1)

    assert_equal(r1, r2)


@pytest.mark.parametrize("ci_method", ["percentile", "bca"])
def test_confidence_interval(ci_method, rng):
    data = rng.normal(size=1000)
    par = stats.norm.fit(data)
    dist = stats.norm(
        par[0], par[1] / len(data) ** 0.5
    )  # accuracy of mean is sqrt(n) better
    cl = 0.9
    ci_ref = dist.ppf(0.05), dist.ppf(0.95)
    ci = confidence_interval(
        np.mean, data, cl=cl, size=1000, ci_method=ci_method, random_state=rng
    )
    assert_allclose(ci_ref, ci, atol=6e-3)


def test_confidence_interval_invalid_p_raises():
    msg = "must be between zero and one"
    with pytest.raises(ValueError, match=msg):
        confidence_interval(np.mean, (1, 2, 3), cl=2)


def test_confidence_interval_invalid_ci_method_raises():
    msg = "method must be 'percentile' or 'bca'"
    with pytest.raises(ValueError, match=msg):
        confidence_interval(np.mean, (1, 2, 3), ci_method="foobar")


def test_bca_confidence_interval_estimator_returns_int(rng):
    def fn(data):
        return int(np.mean(data))

    data = (1, 2, 3)
    ci = confidence_interval(fn, data, ci_method="bca", size=5, random_state=rng)
    assert_allclose((1.0, 2.0), ci)


@pytest.mark.parametrize("ci_method", ["percentile", "bca"])
def test_bca_confidence_interval_bounded_estimator(ci_method, rng):
    def fn(data):
        return max(np.mean(data), 0)

    data = (-3, -2, -1)
    ci = confidence_interval(fn, data, ci_method=ci_method, size=5, random_state=rng)
    assert_allclose((0.0, 0.0), ci)


@pytest.mark.parametrize("method", NON_PARAMETRIC)
def test_variance(method, rng):
    data = np.arange(100)
    v = np.var(data) / len(data)

    r = variance(np.mean, data, size=1000, method=method, random_state=rng)
    assert r == pytest.approx(v, rel=0.05)


@pytest.mark.parametrize("method", NON_PARAMETRIC)
def test_covariance(method, rng):
    cov = np.array([[1.0, 0.1], [0.1, 2.0]])
    data = rng.multivariate_normal([0.1, 0.2], cov, size=1000)

    r = covariance(
        lambda x: np.mean(x, axis=0), data, size=1000, method=method, random_state=rng
    )
    assert_allclose(r, cov / len(data), atol=1e-3)


def test_resample_deprecation(rng):
    data = [1, 2, 3]

    with pytest.warns(FutureWarning):
        r = list(resample(data, 10))
        assert np.shape(r) == (10, 3)

    with pytest.warns(FutureWarning):
        resample(data, 10, "balanced")

    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError):
            resample(data, 10, "foo")

    with pytest.warns(FutureWarning):
        resample(data, 10, "balanced", [1, 1, 2])

    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError):
            resample(data, 10, "balanced", [1, 1])

    with pytest.warns(FutureWarning):
        resample(data, 10, "balanced", [1, 1, 2], rng)

    with pytest.warns(FutureWarning):
        resample(data, 10, "balanced", [1, 1, 2], 1)

    with pytest.warns(FutureWarning):
        with pytest.raises(TypeError):
            resample(data, 10, "balanced", [1, 1, 2], 1.3)

    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError):  # too many arguments
            resample(data, 10, "balanced", [1, 1, 2], 1, 2)


def test_confidence_interval_deprecation(rng):
    d = [1, 2, 3]
    with pytest.warns(FutureWarning):
        r = confidence_interval(np.mean, d, 0.6, random_state=1)
    assert_equal(r, confidence_interval(np.mean, d, cl=0.6, random_state=1))

    with pytest.warns(FutureWarning):
        r = confidence_interval(np.mean, d, 0.6, "percentile", random_state=1)
    assert_equal(
        r,
        confidence_interval(np.mean, d, cl=0.6, ci_method="percentile", random_state=1),
    )

    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError):
            confidence_interval(np.mean, d, 0.6, "percentile", 1)


def test_random_state():
    d = [1, 2, 3]
    a = list(resample(d, size=5, random_state=np.random.default_rng(1)))
    b = list(resample(d, size=5, random_state=1))
    c = list(resample(d, size=5, random_state=[2, 3]))
    assert_equal(a, b)
    assert not np.all([np.all(ai == ci) for (ai, ci) in zip(a, c)])

    with pytest.raises(TypeError):
        resample(d, size=5, random_state=1.5)


@pytest.mark.parametrize("method", NON_PARAMETRIC)
def test_resample_several_args(method):
    a = [1, 2, 3]
    b = [(1, 2), (2, 3), (3, 4)]
    c = ["12", "3", "4"]
    r1 = [[], [], []]
    for ai, bi, ci in resample(a, b, c, size=5, method=method, random_state=1):
        r1[0].append(ai)
        r1[1].append(bi)
        r1[2].append(ci)

    r2 = [[], [], []]
    abc = np.empty(3, dtype=[("a", "i"), ("b", "i", 2), ("c", "U4")])
    abc[:]["a"] = a
    abc[:]["b"] = b
    abc[:]["c"] = c
    for abci in resample(abc, size=5, method=method, random_state=1):
        r2[0].append(abci["a"])
        r2[1].append(abci["b"])
        r2[2].append(abci["c"])

    for i in range(3):
        assert_equal(r1[i], r2[i])


def test_resample_several_args_incompatible_keywords():
    a = [1, 2, 3]
    b = [(1, 2), (2, 3), (3, 4)]
    with pytest.raises(ValueError):
        resample(a, b, size=5, method="norm")

    resample(a, size=5, strata=[1, 1, 2])

    with pytest.raises(ValueError):
        resample(a, b, size=5, strata=[1, 1, 2])

    resample(a, b, a, b, size=5)

    with pytest.raises(ValueError):
        resample(a, [1, 2])

    with pytest.raises(ValueError):
        resample(a, [1, 2, 3, 4])

    with pytest.raises(ValueError):
        resample(a, b, 5)


def test_resample_extended_1():
    a = [1, 2, 3]
    bs = list(resample(a, size=100, method="extended", random_state=1))

    # check that lengths of bootstrap samples are poisson distributed
    w, xe = np.histogram([len(b) for b in bs], bins=10, range=(0, 10))
    wm = stats.poisson(len(a)).pmf(xe[:-1]) * np.sum(w)
    t = np.sum((w - wm) ** 2 / wm)
    pvalue = 1 - stats.chi2(len(w)).cdf(t)
    assert pvalue > 0.1


def test_resample_extended_2():
    n = 10
    a = np.arange(n)
    ts = []
    for b in resample(a, size=1000, method="extended", random_state=1):
        ts.append(np.mean(b))

    t = np.var(ts)
    expected_not_extended = np.var(a) / n

    k = np.arange(100)
    pk = stats.poisson(n).pmf(k)
    expected = expected_not_extended * np.sum(pk[1:] * n / k[1:]) / (1 - pk[0])

    assert expected / expected_not_extended > 1.1
    assert t > expected_not_extended
    assert_allclose(t, expected, atol=0.02)


def test_resample_extended_3():
    n = 10
    a = np.arange(n)
    b = 5 + a
    ns = []
    for ai, bi in resample(a, b, size=1000, method="extended", random_state=1):
        assert len(ai) == len(bi)
        assert_equal(bi - ai, 5)
        ns.append(len(ai))
    assert_allclose(np.var(ns), 10, rtol=0.05)


def test_resample_extended_4():
    x = np.ones(10)
    a = np.transpose((x, 3 * x))

    ts = []
    for b in resample(a, size=1000, method="extended", random_state=1):
        ts.append(np.sum(b, axis=0))

    t = np.var(ts, axis=0)

    mu = np.sum(x, axis=0)
    assert_allclose(t, (mu, 3**2 * mu), rtol=0.05)


def test_resample_extended_5():
    x = np.ones(10)
    a = np.transpose((x, 3 * x))

    ts1 = []
    ts2 = []
    for b1, b2 in resample(a, 3 * a, size=1000, method="extended", random_state=1):
        ts1.append(np.sum(b1, axis=0))
        ts2.append(np.sum(b2, axis=0))

    t1 = np.var(ts1, axis=0)
    t2 = np.var(ts2, axis=0)

    mu1 = np.sum(x, axis=0)
    mu2 = 3**2 * np.sum(x, axis=0)
    assert_allclose(t1, (mu1, 3**2 * mu1), rtol=0.05)
    assert_allclose(t2, (mu2, 3**2 * mu2), rtol=0.05)


def test_bias_error():
    with pytest.raises(NotImplementedError):
        from resample.bootstrap import bias  # noqa

    with pytest.raises(NotImplementedError):
        import resample.bootstrap as b

        b.bias_corrected  # noqa
