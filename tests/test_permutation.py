import numpy as np
from numpy.testing import assert_allclose
from resample import permutation as perm
from resample import _util
from scipy import stats
import pytest


@pytest.fixture
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
        "<TestResult " "statistic=1 pvalue=2 " "samples=[0, 1, 2, ..., 7, 8, 9]>"
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
        got = test(a, b, max_size=500, random_state=1)
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
        got = test(a, b, max_size=500, random_state=1)
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
        got = test(a, b, c, max_size=500, random_state=1)
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
        got = test(a, b, c, max_size=500, random_state=1)
        assert_allclose(expected[0], got[0])
        assert_allclose(expected[1], got[1], atol=5e-2)


def test_bad_input():
    with pytest.raises(ValueError):
        perm.ttest([1, 2, 3], [1.0, np.nan, 2.0])


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


@pytest.mark.parametrize("prec", (0, 0.05, 0.005))
def test_precision(prec, rng):
    x = rng.normal(0, 1, size=100)
    y = rng.normal(0, 2, size=100)

    r = perm.ttest(
        x, y, precision=prec, max_size=10000 if prec > 0 else 123, random_state=1
    )
    if prec == 0:
        assert len(r.samples) == 123
    else:
        n = len(r.samples)
        _, interval = _util.wilson_score_interval(r.pvalue * n, n, 1)
        actual_precision = (interval[1] - interval[0]) / 2
        assert_allclose(actual_precision, prec, atol=0.5 * prec)
