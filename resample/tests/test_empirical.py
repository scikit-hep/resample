import pytest
import numpy as np
from numpy.testing import assert_equal

from resample.empirical import quantile_fn, cdf_fn, influence


# high-quality platform-independent reproducible sequence of pseudo-random numbers
@pytest.fixture
def rng():
    return np.random.Generator(np.random.PCG64(1))


def test_cdf_increasing(rng):
    x = rng.normal(size=100)
    cdf = cdf_fn(x)
    result = [cdf(s) for s in np.linspace(x.min(), x.max(), 100)]
    assert np.all(np.diff(result) >= 0)


def test_cdf_at_infinity():
    cdf = cdf_fn(np.arange(10))
    assert cdf(-np.inf) == 0.0
    assert cdf(np.inf) == 1.0


def test_cdf_simple_cases():
    cdf = cdf_fn([0, 1, 2, 3])
    assert cdf(0) == 0.25
    assert cdf(1) == 0.5
    assert cdf(2) == 0.75
    assert cdf(3) == 1.0


def test_quantile_simple_cases():
    q = quantile_fn([0, 1, 2, 3])
    assert q(0.25) == 0
    assert q(0.5) == 1
    assert q(0.75) == 2
    assert q(1.0) == 3


def test_quantile_is_inverse_of_cdf(rng):
    data = rng.normal(size=100)
    cdf = cdf_fn(data)

    y = cdf(data)

    quant = quantile_fn(data)
    assert_equal([quant(yi) for yi in y], data)  # TODO: quant should be vectorized


@pytest.mark.parametrize("arg", [-1, 1.5])
def test_quantile_out_of_bounds_raises(arg):
    q = quantile_fn(np.array([0, 1, 2, 3]))
    msg = "Argument must be between zero and one"
    with pytest.raises(ValueError, match=msg):
        q(arg)


def test_influence_shape():
    n = 100
    data = np.random.random(n)
    emp = influence(np.mean, data)
    assert len(emp) == n
