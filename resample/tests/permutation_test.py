import numpy as np
from resample.permutation import (ttest,
                                  anova)

np.random.seed(2357)


def test_t_sq_eq_f():
    x = np.random.randn(100)
    y = np.random.randn(100)
    tsq = ttest(x, y)["t"]**2
    f = anova(x, y)["f"]
    assert np.isclose(tsq, f)
