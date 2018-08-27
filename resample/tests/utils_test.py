import pytest
import numpy as np
from resample.utils import ecdf, mise, sup_norm

np.random.seed(2357)

x = np.random.randn(100)
f = ecdf(x)


def test_ecdf_incr():
    result = [f(s) for s in np.linspace(x.min(), x.max(), 100)]
    assert np.all(np.diff(result) >= 0)


def test_ecdf_neg_inf():
    assert f(-np.inf) == 0.0


def test_ecdf_pos_inf():
    assert f(np.inf) == 1.0


def test_ecdf_simple_cases():
    g = ecdf([0, 1, 2, 3])
    assert g(0) == 0.25
    assert g(1) == 0.5
    assert g(2) == 0.75
    assert g(3) == 1.0


def test_mise_inv_domain():
    with pytest.raises(ValueError):
        mise(abs, abs, (1, 0))


def test_mise_eq_func():
    assert mise(abs, abs, (-3, 3)) == 0.0


def test_sup_norm_inv_domain():
    with pytest.raises(ValueError):
        sup_norm(abs, abs, (1, 0))


def test_sup_norm_eq_func():
    assert sup_norm(abs, abs, (-3, 3)) == 0.0
