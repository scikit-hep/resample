import numpy as np
import pytest

from resample.permutation import usp


@pytest.mark.parametrize("n", (10, 100, 1000, 10000))
@pytest.mark.parametrize("k", (2, 10, 100))
@pytest.mark.parametrize("method", ("patefield", "shuffle"))
def test_usp(k, n, method, benchmark):
    w = np.zeros((k, k))
    rng = np.random.default_rng(1)
    for _ in range(n):
        i = rng.integers(k)
        j = rng.integers(k)
        w[i, j] += 1
    assert np.sum(w) == n
    benchmark(lambda: usp(w, method=method, size=100, random_state=1))
