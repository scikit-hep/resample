import numpy as np
import pytest

from scipy.stats import random_table


@pytest.mark.parametrize("n", (10, 100, 1000, 10000, 100000))
@pytest.mark.parametrize("k", (2, 4, 10, 20, 40, 100))
@pytest.mark.parametrize("method", (None, "boyett", "patefield"))
def test_rcont(k, n, method, benchmark):
    w = np.zeros((k, k))
    rng = np.random.default_rng(1)
    for _ in range(n):
        i = rng.integers(k)
        j = rng.integers(k)
        w[i, j] += 1
    r = np.sum(w, axis=1)
    c = np.sum(w, axis=0)
    assert np.sum(r) == n
    assert np.sum(c) == n
    benchmark(lambda: random_table(r, c).rvs(100, method=method, random_state=rng))
