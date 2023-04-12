# ruff: noqa: D100 D103
import numpy as np
import pytest
from numpy.testing import assert_equal

from resample.empirical import cdf_gen, influence, quantile_function_gen


# high-quality platform-independent reproducible sequence of pseudo-random numbers
@pytest.fixture
def rng():
    return np.random.Generator(np.random.PCG64(1))


def test_cdf_increasing(rng):
    x = rng.normal(size=100)
    cdf = cdf_gen(x)
    result = [cdf(s) for s in np.linspace(x.min(), x.max(), 100)]
    assert np.all(np.diff(result) >= 0)


def test_cdf_at_infinity():
    cdf = cdf_gen(np.arange(10))
    assert cdf(-np.inf) == 0.0
    assert cdf(np.inf) == 1.0


def test_cdf_simple_cases():
    cdf = cdf_gen([0, 1, 2, 3])
    assert cdf(0) == 0.25
    assert cdf(1) == 0.5
    assert cdf(2) == 0.75
    assert cdf(3) == 1.0


def test_cdf_on_array():
    x = np.arange(4)
    cdf = cdf_gen(x)
    assert_equal(cdf(x), (x + 1) / len(x))
    assert_equal(cdf(x + 1e-10), (x + 1) / len(x))
    assert_equal(cdf(x - 1e-10), x / len(x))


def test_quantile_simple_cases():
    q = quantile_function_gen([0, 1, 2, 3])
    assert q(0.25) == 0
    assert q(0.5) == 1
    assert q(0.75) == 2
    assert q(1.0) == 3


def test_quantile_on_array():
    x = np.arange(4)
    q = quantile_function_gen(x)
    prob = (x + 1) / len(x)
    assert_equal(q(prob), x)


def test_quantile_is_inverse_of_cdf(rng):
    x = rng.normal(size=30)
    y = cdf_gen(x)(x)
    assert_equal(quantile_function_gen(x)(y), x)


@pytest.mark.parametrize("arg", [-1, 1.5])
def test_quantile_out_of_bounds_is_nan(arg):
    q = quantile_function_gen(np.array([0, 1, 2, 3]))
    assert np.isnan(q(arg))


def test_influence_shape():
    n = 100
    data = np.random.random(n)
    emp = influence(np.mean, data)
    assert len(emp) == n
