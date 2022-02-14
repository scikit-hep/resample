from resample._util import rcont
import numpy as np
from numpy.testing import assert_equal
import pytest


@pytest.mark.parametrize("method", (1, 2))
def test_rcont(method):
    m = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    r = np.sum(m, axis=1)
    c = np.sum(m, axis=0)

    rng = np.random.default_rng(1)
    for w in rcont(5, r, c, method, rng):
        assert_equal(np.sum(w, axis=0), c)
        assert_equal(np.sum(w, axis=1), r)
