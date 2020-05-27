import numpy as np

from resample.permutation import anova, kruskal_wallis, ks_test, ttest, wilcoxon


def test_t_squared_equals_f():
    x = np.random.randn(100)
    y = np.random.randn(100)
    tsq = ttest(x, y)["t"] ** 2
    f = anova([x, y])["f"]
    assert np.isclose(tsq, f)


def test_ks_separable_data():
    x = np.arange(10)
    y = np.arange(10, 20)
    result = ks_test(x, y)
    d = result["d"]
    prop = result["prop"]
    assert np.isclose(d, 1.0)
    assert np.isclose(prop, 0.0)


def test_wilcoxon_separable_data():
    x = np.arange(10)
    y = np.arange(10, 20)
    prop = wilcoxon(x, y)["prop"]
    assert np.isclose(prop, 0.0)


def test_kruskal_wallis_separable_data():
    x = np.arange(10)
    y = np.arange(10, 21)
    prop = kruskal_wallis([x, y])["prop"]
    assert np.isclose(prop, 0.0)
