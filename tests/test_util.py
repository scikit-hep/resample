# ruff: noqa: D100 D103
from resample import _util as u
import numpy as np
from numpy.testing import assert_allclose


def test_wilson_score_interval():
    n = 100
    for n1 in (10, 50, 90):
        p, lh = u.wilson_score_interval(n1, n, 1)
        s = np.sqrt(p * (1 - p) / n)
        assert_allclose(p, n1 / n)
        assert_allclose(lh, (p - s, p + s), atol=0.01)

    n = 10
    n1 = 0
    p, lh = u.wilson_score_interval(n1, n, 1)
    assert_allclose(p, 0.0)
    assert_allclose(lh, (0, 0.1), atol=0.01)

    n1 = 10
    p, lh = u.wilson_score_interval(n1, n, 1)
    assert_allclose(p, 1.0)
    assert_allclose(lh, (0.9, 1.0), atol=0.01)
