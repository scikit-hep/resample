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
