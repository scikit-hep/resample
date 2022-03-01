import numpy as np
from numpy.testing import assert_allclose
from resample import permutation as perm
from resample import _util
import pytest

pytest.importorskip("pyximport")


@pytest.fixture
def rng():
    return np.random.Generator(np.random.PCG64(1))


@pytest.mark.parametrize("method", ("auto", "patefield", "shuffle"))
def test_usp_1(method, rng):
    x = rng.normal(0, 2, size=100)
    y = rng.normal(1, 3, size=100)

    w = np.histogram2d(x, y, bins=(5, 10))[0]
    r = perm.usp(w, max_size=100, random_state=1)
    assert r.pvalue > 0.05


@pytest.mark.parametrize("method", ("auto", "patefield", "shuffle"))
def test_usp_2(method, rng):
    x = rng.normal(0, 2, size=100).astype(int)

    w = np.histogram2d(x, x, range=((-5, 5), (-5, 5)))[0]

    r = perm.usp(w, method=method, max_size=99, random_state=1)
    assert r.pvalue == 0.01


@pytest.mark.parametrize("method", ("auto", "patefield", "shuffle"))
def test_usp_3(method, rng):
    cov = np.empty((2, 2))
    cov[0, 0] = 2 ** 2
    cov[1, 1] = 3 ** 2
    rho = 0.5
    cov[0, 1] = rho * np.sqrt(cov[0, 0] * cov[1, 1])
    cov[1, 0] = cov[0, 1]

    xy = rng.multivariate_normal([0, 1], cov, size=500).astype(int)

    w = np.histogram2d(*xy.T)[0]

    r = perm.usp(w, method=method, random_state=1)
    assert r.pvalue < 0.0012


@pytest.mark.parametrize("method", ("auto", "patefield", "shuffle"))
def test_usp_4(method):
    # table1 from https://doi.org/10.1098/rspa.2021.0549
    w = [[18, 36, 21, 9, 6], [12, 36, 45, 36, 21], [6, 9, 9, 3, 3], [3, 9, 9, 6, 3]]
    r1 = perm.usp(w, precision=0, method=method, max_size=10000, random_state=1)
    r2 = perm.usp(np.transpose(w), method=method, max_size=1, random_state=1)
    assert_allclose(r1.statistic, r2.statistic)
    expected = 0.004106  # checked against USP R package
    assert_allclose(r1.statistic, expected, atol=1e-6)
    # according to paper, pvalue is 0.001, but USP R package gives correct value
    expected = 0.0024  # computed from USP R package with b=99999
    assert_allclose(r1.pvalue, expected, atol=0.001)


@pytest.mark.parametrize("method", ("auto", "patefield", "shuffle"))
def test_usp_5(method, rng):
    w = np.empty((100, 100))
    for i in range(100):
        for j in range(100):
            w[i, j] = (i + j) % 2
    r = perm.usp(w, method=method, max_size=100, random_state=1)
    assert r.pvalue > 0.1


def test_usp_bias(rng):
    # We compute the p-value as an upper limit to the type I error rate.
    # Therefore, the p-value is not unbiased. For max_size=1, we expect
    # an average p-value = (1 + 0.5) / (1 + 1) = 0.75
    got = [
        perm.usp(rng.poisson(1000, size=(2, 2)), max_size=1, random_state=i).pvalue
        for i in range(1000)
    ]
    assert_allclose(np.mean(got), 0.75, atol=0.05)


def test_usp_bad_input():
    with pytest.raises(ValueError):
        perm.usp([[1, 2], [3, 4]], precision=-1)

    with pytest.raises(ValueError):
        perm.usp([[1, 2], [3, 4]], max_size=0)

    with pytest.raises(ValueError):
        perm.usp([[1, 2], [3, 4]], max_size=-1)

    with pytest.raises(ValueError):
        perm.usp([1, 2])

    with pytest.raises(ValueError):
        perm.usp([[1, 2], [3, 4]], method="foo")


@pytest.mark.parametrize("prec", (0, 0.05, 0.005))
def test_precision(prec, rng):
    x = rng.normal(0, 1, size=100)
    y = rng.normal(0, 2, size=100)
    w = np.histogram2d(x, y)[0]

    r = perm.usp(w, precision=prec, max_size=10000 if prec > 0 else 123, random_state=1)
    if prec == 0:
        assert len(r.samples) == 123
    else:
        n = len(r.samples)
        _, interval = _util.wilson_score_interval(r.pvalue * n, n, 1)
        actual_precision = (interval[1] - interval[0]) / 2
        assert_allclose(actual_precision, prec, atol=0.5 * prec)
