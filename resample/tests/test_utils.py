import pytest
import numpy as np
from resample.utils import ecdf, mise, sup_norm


def test_ecdf_increasing():
    x = np.random.randn(100)
    f = ecdf(x)
    result = [f(s) for s in np.linspace(x.min(), x.max(), 100)]
    assert np.all(np.diff(result) >= 0)


def test_ecdf_at_infinity():
    f = ecdf(np.arange(10))
    assert f(-np.inf) == 0.0
    assert f(np.inf) == 1.0


def test_ecdf_simple_cases():
    g = ecdf([0, 1, 2, 3])
    assert g(0) == 0.25
    assert g(1) == 0.5
    assert g(2) == 0.75
    assert g(3) == 1.0


def test_mise_invalid_domain():
    with pytest.raises(ValueError):
        mise(abs, abs, (1, 0))


def test_mise_identical_functions():
    assert mise(abs, abs, (-3, 3)) == 0.0


def test_sup_norm_invalid_domain():
    with pytest.raises(ValueError):
        sup_norm(abs, abs, (1, 0))


def test_sup_norm_identical_functions():
    assert sup_norm(abs, abs, (-3, 3)) == 0.0
