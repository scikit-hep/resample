from resample import _util as u
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest


@pytest.mark.parametrize("method", (0, 1))
def test_rcont_1(method):
    m = np.arange(6).reshape((3, 2))
    r = np.sum(m, axis=1)
    c = np.sum(m, axis=0)

    rng = np.random.default_rng(1)
    for w in u.rcont(5, r, c, method, rng):
        assert_equal(np.sum(w, axis=0), c)
        assert_equal(np.sum(w, axis=1), r)


@pytest.mark.parametrize("shape", ((2, 3), (3, 2), (3, 3), (3, 4), (4, 3)))
def test_rcont_2(shape):
    m = np.arange(np.prod(shape)).reshape(shape)
    r = np.sum(m, axis=1)
    c = np.sum(m, axis=0)

    # Patefield should give same results if zero row or column is inserted

    rng = np.random.default_rng(1)
    w1 = u.rcont(5, r, c, 1, rng)

    r2 = np.zeros(len(r) + 1)
    r2[0] = r[0]
    r2[2:] = r[1:]
    rng = np.random.default_rng(1)
    w2 = u.rcont(5, r2, c, 1, rng)
    assert_equal(w2[:, 1, :], 0)
    mask = np.ones(w2.shape[1], dtype=bool)
    mask[1] = False
    assert_equal(w2[:, mask, :], w1)

    c2 = np.zeros(len(c) + 1)
    c2[0] = c[0]
    c2[2:] = c[1:]
    rng = np.random.default_rng(1)
    w2 = u.rcont(5, r, c2, 1, rng)
    assert_equal(w2[:, :, 1], 0)
    mask = np.ones(w2.shape[2], dtype=bool)
    mask[1] = False
    assert_equal(w2[:, :, mask], w1)

    rng = np.random.default_rng(1)
    w2 = u.rcont(5, r2, c2, 1, rng)
    assert_equal(w2[:, 1, 1], 0)
    r_mask = np.ones(w2.shape[1], dtype=bool)
    r_mask[1] = False
    c_mask = np.ones(w2.shape[2], dtype=bool)
    c_mask[1] = False
    assert_equal(w2[:, r_mask, :][:, :, c_mask], w1)


def test_rcont_bad_method():
    m = np.arange(6).reshape(3, 2)
    r = np.sum(m, axis=1)
    c = np.sum(m, axis=0)
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        u.rcont(5, r, c, 2, rng)


@pytest.mark.parametrize("method", (0, 1))
def test_rcont_bad_input(method):
    m = np.arange(6).reshape(3, 2)
    r = np.sum(m, axis=1)
    c = np.sum(m, axis=0)
    rng = np.random.default_rng(1)

    # wrong dimension
    with pytest.raises(ValueError):
        u.rcont(5, np.arange(4).reshape((2, 2)), c, method, rng)

    # wrong dimension
    with pytest.raises(ValueError):
        u.rcont(5, r, np.arange(4).reshape((2, 2)), method, rng)

    # degenerate table
    with pytest.raises(ValueError):
        u.rcont(5, np.arange(1), np.arange(2), method, rng)

    # degenerate table
    with pytest.raises(ValueError):
        u.rcont(5, np.arange(2), np.arange(1), method, rng)

    # negative entries
    r2 = r.copy()
    r2[0] = -1
    with pytest.raises(ValueError):
        u.rcont(5, r2, c, method, rng)

    # negative entries
    c2 = c.copy()
    c2[-1] = -1
    with pytest.raises(ValueError):
        u.rcont(5, r, c2, method, rng)

    # sum(r) != sum(c)
    with pytest.raises(ValueError):
        u.rcont(5, r + 1, c, method, rng)

    # total is zero
    with pytest.raises(ValueError):
        u.rcont(5, np.zeros(2), np.zeros(3), method, rng)


def test_wilson_score_interval():
    n = 100
    for n1 in (10, 50, 90):
        p, lh = u.wilson_score_interval(n1, n, 1)
        s = np.sqrt(p * (1 - p) / n)
        assert_allclose(p, n1 / n)
        assert_allclose(lh, (p - s, p + s), atol=0.01)

    n = 10
    n1 = 0
    p, lh = u.wilson_score_interval(n1, n, 1)
    assert_allclose(p, 0.0)
    assert_allclose(lh, (0, 0.1), atol=0.01)

    n1 = 10
    p, lh = u.wilson_score_interval(n1, n, 1)
    assert_allclose(p, 1.0)
    assert_allclose(lh, (0.9, 1.0), atol=0.01)
