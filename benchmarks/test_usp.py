import numpy as np
import pytest

from resample.permutation import usp


@pytest.mark.parametrize("m", (2, 10, 100, 1000))
@pytest.mark.parametrize("n", (2, 10, 100, 1000))
def test_usp(m, n, benchmark):
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, size=10000)
    y = rng.normal(0, 1, size=10000)
    w = np.histogram2d(x, y, bins=(m, n))[0]
    benchmark(lambda: usp(w, precision=0, max_size=100))
