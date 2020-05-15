import numpy as np
from resample.permutation import ttest, anova, ks_test


def test_t_squared_equals_f():
    x = np.random.randn(100)
    y = np.random.randn(100)
    tsq = ttest(x, y)["t"] ** 2
    f = anova([x, y])["f"]
    assert np.isclose(tsq, f)


def test_ks_separable_data():
    x = np.random.random(100)
    y = x.max() + x
    d = ks_test(x, y)["d"]
    assert np.isclose(d, 1.0)
