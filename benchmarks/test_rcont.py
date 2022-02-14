import numpy as np
import pytest

from resample._util import rcont


@pytest.mark.parametrize("k", (2, 10, 100))
@pytest.mark.parametrize("n", (10, 100, 1000, 10000, 100000))
@pytest.mark.parametrize("method", (0, 1))
def test_rcont(k, n, method, benchmark):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    w = np.histogram2d(x, y, bins=(k, k))[0]
    r = np.sum(w, axis=1)
    c = np.sum(w, axis=0)
    benchmark(lambda: rcont(100, r, c, method, rng))
