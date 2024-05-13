# ruff: noqa: D100 D101 D103 D105 D107
import numpy as np
from numpy.testing import assert_allclose
from resample import permutation as perm
from scipy import stats
import pytest


@pytest.fixture()
def rng():
    return np.random.Generator(np.random.PCG64(1))


def test_TestResult():
    p = perm.TestResult(1, 2, [3, 4])
    assert p.statistic == 1
    assert p.pvalue == 2
    assert p.samples == [3, 4]
    assert repr(p) == "<TestResult statistic=1 pvalue=2 samples=[3, 4]>"
    assert len(p) == 3
    first, *rest = p
    assert first == 1
    assert rest == [2, [3, 4]]

    p2 = perm.TestResult(1, 2, np.arange(10))
    assert repr(p2) == (
        "<TestResult statistic=1 pvalue=2 samples=[0, 1, 2, ..., 7, 8, 9]>"
    )


class Scipy:
    def __init__(self, **kwargs):
        self.d = kwargs

    def __getitem__(self, key):
        if key in self.d:
            return self.d[key]
        return getattr(stats, key)


scipy = Scipy(
    anova=stats.f_oneway,
    ttest=lambda x, y: stats.ttest_ind(x, y, equal_var=False),
)


@pytest.mark.parametrize(
    "test_name",
    (
        "anova",
        "kruskal",
        "pearsonr",
        "spearmanr",
        "ttest",
    ),
)
@pytest.mark.parametrize("size", (10, 100))
def test_two_sample_same_size(test_name, size, rng):
    x = rng.normal(size=size)
    y = rng.normal(1, size=size)

    test = getattr(perm, test_name)
    scipy_test = scipy[test_name]

    for a, b in ((x, y), (y, x)):
        expected = scipy_test(a, b)
        got = test(a, b, size=999, random_state=1)
        assert_allclose(expected[0], got[0])
        assert_allclose(expected[1], got[1], atol={10: 0.2, 100: 0.02}[size])


@pytest.mark.parametrize(
    "test_name",
    (
        "anova",
        "kruskal",
        "pearsonr",
        "spearmanr",
        "ttest",
    ),
)
@pytest.mark.parametrize("size", (10, 100))
def test_two_sample_different_size(test_name, size, rng):
    x = rng.normal(size=size)
    y = rng.normal(1, size=2 * size)

    test = getattr(perm, test_name)
    scipy_test = scipy[test_name]

    if test_name in ("pearsonr", "spearmanr"):
        with pytest.raises(ValueError):
            test(x, y)
        return

    for a, b in ((x, y), (y, x)):
        expected = scipy_test(a, b)
        got = test(a, b, size=999, random_state=1)
        assert_allclose(expected[0], got[0])
        assert_allclose(expected[1], got[1], atol=5e-2)


@pytest.mark.parametrize(
    "test_name",
    (
        "anova",
        "kruskal",
    ),
)
@pytest.mark.parametrize("size", (10, 100))
def test_three_sample_same_size(test_name, size, rng):
    x = rng.normal(size=size)
    y = rng.normal(1, size=size)
    z = rng.normal(0.5, size=size)

    test = getattr(perm, test_name)
    scipy_test = scipy[test_name]

    for a, b, c in ((x, y, z), (z, y, x)):
        expected = scipy_test(a, b, c)
        got = test(a, b, c, size=999, random_state=1)
        assert_allclose(expected[0], got[0])
        assert_allclose(expected[1], got[1], atol=5e-2)


@pytest.mark.parametrize(
    "test_name",
    (
        "anova",
        "kruskal",
    ),
)
@pytest.mark.parametrize("size", (10, 100))
def test_three_sample_different_size(test_name, size, rng):
    x = rng.normal(size=size)
    y = rng.normal(1, size=2 * size)
    z = rng.normal(0.5, size=size * 2)

    test = getattr(perm, test_name)
    scipy_test = scipy[test_name]

    for a, b, c in ((x, y, z), (z, y, x)):
        expected = scipy_test(a, b, c)
        got = test(a, b, c, size=500, random_state=1)
        assert_allclose(expected[0], got[0])
        assert_allclose(expected[1], got[1], atol=5e-2)


def test_bad_input():
    with pytest.raises(ValueError):
        perm.ttest([1, 2, 3], [1.0, np.nan, 2.0])


@pytest.mark.parametrize("method", ("auto", "patefield", "boyett"))
def test_usp_1(method, rng):
    x = rng.normal(0, 2, size=100)
    y = rng.normal(1, 3, size=100)

    w = np.histogram2d(x, y, bins=(5, 10))[0]
    r = perm.usp(w, method=method, size=100, random_state=1)
    assert r.pvalue > 0.05


@pytest.mark.parametrize("method", ("auto", "patefield", "boyett"))
def test_usp_2(method, rng):
    x = rng.normal(0, 2, size=100).astype(int)

    w = np.histogram2d(x, x, range=((-5, 5), (-5, 5)))[0]

    r = perm.usp(w, method=method, size=99, random_state=1)
    assert r.pvalue == 0.01


@pytest.mark.parametrize("method", ("auto", "patefield", "boyett"))
def test_usp_3(method, rng):
    cov = np.empty((2, 2))
    cov[0, 0] = 2**2
    cov[1, 1] = 3**2
    rho = 0.5
    cov[0, 1] = rho * np.sqrt(cov[0, 0] * cov[1, 1])
    cov[1, 0] = cov[0, 1]

    xy = rng.multivariate_normal([0, 1], cov, size=500).astype(int)

    w = np.histogram2d(*xy.T)[0]

    r = perm.usp(w, method=method, random_state=1)
    assert r.pvalue < 0.0012


@pytest.mark.parametrize("method", ("auto", "patefield", "boyett"))
def test_usp_4(method):
    # table1 from https://doi.org/10.1098/rspa.2021.0549
    w = [[18, 36, 21, 9, 6], [12, 36, 45, 36, 21], [6, 9, 9, 3, 3], [3, 9, 9, 6, 3]]
    r1 = perm.usp(w, method=method, size=9999, random_state=1)
    r2 = perm.usp(np.transpose(w), method=method, size=1, random_state=1)
    assert_allclose(r1.statistic, r2.statistic)
    expected = 0.004106  # checked against USP R package
    assert_allclose(r1.statistic, expected, atol=1e-6)
    # according to paper, pvalue is 0.001, but USP R package gives correct value
    expected = 0.0024  # computed from USP R package with b=99999
    assert_allclose(r1.pvalue, expected, atol=0.001)


@pytest.mark.parametrize("method", ("auto", "patefield", "boyett"))
def test_usp_5(method, rng):
    w = np.empty((100, 100))
    for i in range(100):
        for j in range(100):
            w[i, j] = (i + j) % 2
    r = perm.usp(w, method=method, size=99, random_state=1)
    assert r.pvalue > 0.1


def test_usp_bias(rng):
    # We compute the p-value as an upper limit to the type I error rate.
    # Therefore, the p-value is not unbiased. For size=1, we expect
    # an average p-value = (1 + 0.5) / (1 + 1) = 0.75
    got = [
        perm.usp(rng.poisson(1000, size=(2, 2)), size=1, random_state=i).pvalue
        for i in range(1000)
    ]
    assert_allclose(np.mean(got), 0.75, atol=0.05)


def test_usp_bad_input():
    with pytest.raises(ValueError):
        perm.usp([[1, 2], [3, 4]], size=0)

    with pytest.raises(ValueError):
        perm.usp([[1, 2], [3, 4]], size=-1)

    with pytest.raises(ValueError):
        perm.usp([1, 2])

    with pytest.raises(ValueError):
        perm.usp([[1, 2], [3, 4]], method="foo")


def test_usp_deprecrated():
    w = [[1, 2, 3], [4, 5, 6]]
    r1 = perm.usp(w, method="boyett", size=100, random_state=1)
    with pytest.warns(FutureWarning):
        r2 = perm.usp(w, method="shuffle", size=100, random_state=1)
    assert r1.statistic == r2.statistic


def test_ttest_bad_input():
    with pytest.raises(ValueError):
        perm.ttest([1, 2], [3, 4], size=0)

    with pytest.raises(ValueError):
        perm.ttest([1, 2], [3, 4], size=-1)

    with pytest.raises(ValueError):
        perm.ttest(1, 2)

    with pytest.raises(ValueError):
        perm.ttest([1], [2])
