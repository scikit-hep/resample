import numpy as np
from numpy.testing import assert_allclose
from resample import permutation as perm
from scipy import stats
import pytest


@pytest.fixture
def rng():
    return np.random.Generator(np.random.PCG64(1))


def test_TestResult():
    p = perm.TestResult(1, 2, (1, 3), [3, 4])
    assert p.statistic == 1
    assert p.pvalue == 2
    assert p.interval == (1, 3)
    assert p.samples == [3, 4]
    assert repr(p) == "<TestResult statistic=1 pvalue=2 interval=(1, 3) samples=[3, 4]>"
    assert len(p) == 3
    first, *rest = p
    assert first == 1
    assert rest == [2, (1, 3), [3, 4]]

    p2 = perm.TestResult(1, 2, (1, 3), np.arange(10))
    assert repr(p2) == (
        "<TestResult "
        "statistic=1 pvalue=2 interval=(1, 3) "
        "samples=[0, 1, 2, ..., 7, 8, 9]>"
    )


def test_wilson_score_interval():
    n = 100
    for n1 in (10, 50, 90):
        p, lh = perm._wilson_score_interval(n1, n, 1)
        s = np.sqrt(p * (1 - p) / n)
        assert_allclose(p, n1 / n)
        assert_allclose(lh, (p - s, p + s), atol=0.01)

    n = 10
    n1 = 0
    p, lh = perm._wilson_score_interval(n1, n, 1)
    assert_allclose(p, 0.0)
    assert_allclose(lh, (0, 0.1), atol=0.01)

    n1 = 10
    p, lh = perm._wilson_score_interval(n1, n, 1)
    assert_allclose(p, 1.0)
    assert_allclose(lh, (0.9, 1.0), atol=0.01)


scipy = {
    "anova": stats.f_oneway,
    "mannwhitneyu": lambda x, y: stats.mannwhitneyu(x, y, alternative="two-sided"),
    "kruskal": stats.kruskal,
    "ks": stats.ks_2samp,
    "pearson": stats.pearsonr,
    "spearman": stats.spearmanr,
    "ttest": lambda x, y: stats.ttest_ind(x, y, equal_var=False),
}


@pytest.mark.parametrize(
    "test_name",
    (
        "anova",
        "mannwhitneyu",
        "kruskal",
        "ks",
        "pearson",
        "spearman",
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
        got = test(a, b, random_state=1)
        assert_allclose(expected[0], got[0])
        assert_allclose(expected[1], got[1], atol={10: 0.2, 100: 0.02}[size])


@pytest.mark.parametrize(
    "test_name",
    (
        "anova",
        "mannwhitneyu",
        "kruskal",
        "ks",
        "pearson",
        "spearman",
        "ttest",
    ),
)
@pytest.mark.parametrize("size", (10, 100))
def test_two_sample_different_size(test_name, size, rng):
    x = rng.normal(size=size)
    y = rng.normal(1, size=2 * size)

    test = getattr(perm, test_name)
    scipy_test = scipy[test_name]

    if test_name in ("pearson", "spearman"):
        with pytest.raises(ValueError):
            test(x, y)
        return

    for a, b in ((x, y), (y, x)):
        expected = scipy_test(a, b)
        got = test(a, b, random_state=1)
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
        got = test(a, b, c, random_state=1)
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
        got = test(a, b, c, random_state=1)
        assert_allclose(expected[0], got[0])
        assert_allclose(expected[1], got[1], atol=5e-2)


def test_bad_input():
    with pytest.raises(ValueError):
        perm.ttest([1, 2, 3], [1.0, np.nan, 2.0])


def test_usp_1(rng):
    x = rng.normal(0, 2, size=100)
    y = rng.normal(1, 3, size=100)

    w = np.histogram2d(x, y)[0]

    r = perm.usp(w, max_size=100, random_state=1)
    assert r.pvalue > 0.05


def test_usp_2(rng):
    x = rng.normal(0, 2, size=100).astype(int)

    w = np.histogram2d(x, x)[0]

    r = perm.usp(w, max_size=100, random_state=1)
    assert r.pvalue == 0


def test_usp_3(rng):
    cov = np.empty((2, 2))
    cov[0, 0] = 2**2
    cov[1, 1] = 3**2
    rho = 0.5
    cov[0, 1] = rho * np.sqrt(cov[0, 0] * cov[1, 1])
    cov[1, 0] = cov[0, 1]

    xy = rng.multivariate_normal([0, 1], cov, size=500).astype(int)

    w = np.histogram2d(*xy.T)[0]

    r = perm.usp(w, random_state=1)
    assert r.pvalue < 0.001


def test_usp_4():
    # table1 from https://doi.org/10.1098/rspa.2021.0549
    w = [[18, 36, 21, 9, 6], [12, 36, 45, 36, 21], [6, 9, 9, 3, 3], [3, 9, 9, 6, 3]]
    r = perm.usp(w, precision=0, max_size=1000, random_state=1)
    # according to paper, pvalue is 0.001, but we get 0.0025 in high-statistics runs
    assert_allclose(r.pvalue, 0.0025, atol=0.001)
    _, interval = perm._wilson_score_interval(0.0025 * 1000, 1000, 1)
    assert_allclose(r.interval, interval, atol=0.001)


def test_usp_bad_input():
    with pytest.raises(ValueError):
        perm.usp([[1, 2], [3, 4]], precision=-1)

    with pytest.raises(ValueError):
        perm.usp([[1, 2], [3, 4]], max_size=0)

    with pytest.raises(ValueError):
        perm.usp([[1, 2], [3, 4]], max_size=-1)

    with pytest.raises(ValueError):
        perm.usp([1, 2])


def test_ttest_bad_input():
    with pytest.raises(ValueError):
        perm.ttest([1, 2], [3, 4], precision=-1)

    with pytest.raises(ValueError):
        perm.ttest([1, 2], [3, 4], max_size=0)

    with pytest.raises(ValueError):
        perm.ttest([1, 2], [3, 4], max_size=-1)

    with pytest.raises(ValueError):
        perm.ttest(1, 2)

    with pytest.raises(ValueError):
        perm.ttest([1], [2])


@pytest.mark.parametrize("test", (perm.ttest, perm.usp))
@pytest.mark.parametrize("prec", (0, 0.05, 0.005))
def test_precision_1(test, prec, rng):
    x = rng.normal(0, 1, size=100)
    y = rng.normal(0, 2, size=100)
    if test is perm.ttest:
        args = (x, y)
    else:
        w = np.histogram2d(x, y)[0]
        args = (w,)

    r = test(*args, precision=prec, max_size=10000 if prec > 0 else 123)
    if prec == 0:
        assert len(r.samples) == 123
    else:
        assert (r.interval[1] - r.interval[0]) / 2 < prec
        assert_allclose((r.interval[1] - r.interval[0]) / 2, prec, atol=0.5 * prec)
