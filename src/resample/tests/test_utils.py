import pytest
import numpy as np
from resample.utils import ecdf, eqf, mise, sup_norm


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
    g = ecdf(np.array([0, 1, 2, 3]))
    assert g(0) == 0.25
    assert g(1) == 0.5
    assert g(2) == 0.75
    assert g(3) == 1.0


def test_eqf_simple_cases():
    g = eqf(np.array([0, 1, 2, 3]))
    assert g(0.25) == 0
    assert g(0.5) == 1
    assert g(0.75) == 2
    assert g(1.0) == 3


@pytest.mark.parametrize("arg", [-1, 1.5])
def test_eqf_out_of_bounds_raises(arg):
    g = eqf(np.array([0, 1, 2, 3]))
    msg = "Argument must be between zero and one"
    with pytest.raises(ValueError, match=msg):
        g(arg)


def test_mise_invalid_domain():
    msg = "Invalid domain"
    with pytest.raises(ValueError, match=msg):
        mise(abs, abs, (1, 0))


def test_mise_identical_functions():
    assert mise(abs, abs, (-3, 3)) == 0.0


def test_sup_norm_invalid_domain():
    msg = "Invalid domain"
    with pytest.raises(ValueError, match=msg):
        sup_norm(abs, abs, (1, 0))


def test_sup_norm_identical_functions():
    assert sup_norm(abs, abs, (-3, 3)) == 0.0
