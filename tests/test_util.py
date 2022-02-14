from resample._util import rcont
import numpy as np
from numpy.testing import assert_equal
import pytest


@pytest.mark.parametrize("method", (1, 2))
def test_rcont_1(method):
    m = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    r = np.sum(m, axis=1)
    c = np.sum(m, axis=0)

    rng = np.random.default_rng(1)
    for w in rcont(5, r, c, method, rng):
        assert_equal(np.sum(w, axis=0), c)
        assert_equal(np.sum(w, axis=1), r)


def test_rcont_2():
    m = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    r = np.sum(m, axis=1)
    c = np.sum(m, axis=0)

    # Patefield should give same results if zero row or column is inserted

    rng = np.random.default_rng(1)
    w1 = rcont(5, r, c, 1, rng)

    r2 = np.zeros(len(r) + 1)
    r2[0] = r[0]
    r2[2:] = r[1:]
    rng = np.random.default_rng(1)
    w2 = rcont(5, r2, c, 1, rng)
    assert_equal(w2[:, 1, :], 0)
    mask = np.ones(w2.shape[1], dtype=bool)
    mask[1] = False
    assert_equal(w2[:, mask, :], w1)

    c2 = np.zeros(len(c) + 1)
    c2[0] = c[0]
    c2[2:] = c[1:]
    rng = np.random.default_rng(1)
    w2 = rcont(5, r, c2, 1, rng)
    assert_equal(w2[:, :, 1], 0)
    mask = np.ones(w2.shape[2], dtype=bool)
    mask[1] = False
    assert_equal(w2[:, :, mask], w1)

    rng = np.random.default_rng(1)
    w2 = rcont(5, r2, c2, 1, rng)
    assert_equal(w2[:, 1, 1], 0)
    r_mask = np.ones(w2.shape[1], dtype=bool)
    r_mask[1] = False
    c_mask = np.ones(w2.shape[2], dtype=bool)
    c_mask[1] = False
    assert_equal(w2[:, r_mask, :][:, :, c_mask], w1)
