import numpy as np
from numpy.testing import assert_allclose
from resample import permutation as perm
from scipy import stats
import pytest


@pytest.fixture
def rng():
    return np.random.Generator(np.random.PCG64(1))


def test_PermutationResult():
    p = perm.PermutationResult(1, 2, [3, 4])
    assert p.statistic == 1
    assert p.pvalue == 2
    assert p.samples == [3, 4]
    assert repr(p) == "<PermutationResult statistic=1 pvalue=2 samples=[3, 4]>"
    assert len(p) == 3
    first, *rest = p
    assert first == 1
    assert rest == [2, [3, 4]]


def test_ttest_statistic(rng):
    expected = []
    got = []
    for size in range(2, 100):
        x = rng.normal(size=size)
        y = rng.normal(1, size=size * 2)
        t = stats.ttest_ind(x, y, equal_var=False).statistic
        expected.append(t)
        t = perm._ttest([x, y])
        got.append(t)
    assert_allclose(expected, got)


def test_anova_statistic(rng):
    expected = []
    got = []
    for size in range(2, 100):
        x = rng.normal(size=size)
        y = rng.normal(1, size=size * 2)
        z = rng.normal(0.5, size=size * 2)
        t = stats.f_oneway(x, y, z).statistic
        expected.append(t)
        anova = perm._ANOVA()
        t = anova([x, y, z])
        got.append(t)
    assert_allclose(expected, got)


def test_mannwhitneyu_statistic(rng):
    got = []
    expected = []
    for size in range(1, 100):
        x = rng.normal(size=size)
        y = rng.normal(1, size=size)
        got.append(perm._mannwhitneyu([x, y]))
        expected.append(stats.mannwhitneyu(x, y, alternative="two-sided").statistic)
    assert_allclose(expected, got)


def test_kruskal_statistic(rng):
    got = []
    expected = []
    for size in range(2, 100):
        x = rng.normal(size=size)
        y = rng.normal(1, size=size)
        z = rng.normal(0.5, size=size)
        got.append(perm._kruskal([x, y, z]))
        expected.append(stats.kruskal(x, y, z)[0])
    assert_allclose(expected, got)


def test_pearson_statistic(rng):
    got = []
    expected = []
    for size in range(2, 100):
        x = rng.normal(size=size)
        y = rng.normal(1, size=size)
        got.append(perm._pearson([x, y]))
        expected.append(stats.pearsonr(x, y)[0])
    assert_allclose(expected, got)


def test_spearman_statistic(rng):
    got = []
    expected = []
    for size in range(2, 100):
        x = rng.normal(size=size)
        y = rng.normal(1, size=size)
        got.append(perm._spearman([x, y]))
        expected.append(stats.spearmanr(x, y)[0])
    assert_allclose(expected, got)


def test_ks_statistic(rng):
    got = []
    expected = []
    for size in range(2, 100):
        x = rng.normal(size=size)
        y = rng.normal(1, size=size)
        ks = perm._KS()
        got.append(ks([x, y]))
        expected.append(stats.ks_2samp(x, y)[0])
    assert_allclose(expected, got)


def test_ttest_separable_data(rng):
    x = np.arange(10)
    y = np.arange(10, 20)
    r = perm.ttest(x, y, random_state=rng)
    assert_allclose(r.pvalue, 0.0)


def test_mannwhitneyu_separable_data():
    x = np.arange(10)
    y = np.arange(10, 20)
    r = perm.mannwhitneyu(x, y, random_state=1)
    assert_allclose(r.pvalue, 0.0)


def test_pearson_separable_data(rng):
    x = np.arange(10)
    y = np.arange(10, 20)
    r = perm.pearson(x, y, random_state=rng)
    assert_allclose(r.pvalue, 0.0)


def test_spearman_separable_data(rng):
    x = np.arange(10)
    y = np.arange(10, 20)
    r = perm.spearman(x, y, random_state=rng)
    assert_allclose(r.pvalue, 0.0)


def test_ks_separable_data(rng):
    x = np.arange(10)
    y = np.arange(10, 20)
    r = perm.ks(x, y, random_state=rng)
    assert_allclose(r.pvalue, 0.0)


def test_kruskal_separable_data():
    x = np.arange(10)
    y = np.arange(10, 20)
    r = perm.kruskal([x, y], random_state=1)
    assert_allclose(r.pvalue, 0.0)
