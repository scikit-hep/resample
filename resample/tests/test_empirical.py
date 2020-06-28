import pytest
import numpy as np

from resample.empirical import quantile, cdf


def test_cdf_increasing():
    x = np.random.randn(100)
    f = cdf(x)
    result = [f(s) for s in np.linspace(x.min(), x.max(), 100)]
    assert np.all(np.diff(result) >= 0)


def test_cdf_at_infinity():
    f = cdf(np.arange(10))
    assert f(-np.inf) == 0.0
    assert f(np.inf) == 1.0


def test_cdf_simple_cases():
    g = cdf(np.array([0, 1, 2, 3]))
    assert g(0) == 0.25
    assert g(1) == 0.5
    assert g(2) == 0.75
    assert g(3) == 1.0


def test_quantile_simple_cases():
    g = quantile(np.array([0, 1, 2, 3]))
    assert g(0.25) == 0
    assert g(0.5) == 1
    assert g(0.75) == 2
    assert g(1.0) == 3


@pytest.mark.parametrize("arg", [-1, 1.5])
def test_quantile_out_of_bounds_raises(arg):
    g = quantile(np.array([0, 1, 2, 3]))
    msg = "Argument must be between zero and one"
    with pytest.raises(ValueError, match=msg):
        g(arg)


def test_empirical_influence_shape():
    n = 100
    arr = np.random.random(n)
    emp = empirical_influence(arr, np.mean)
    assert len(emp) == n
