import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest

from resample.jackknife import (
    resample,
    jackknife,
    bias,
    bias_corrected,
    variance,
)


def test_resample_1d():
    data = (1, 2, 3)

    r = []
    for x in resample(data):
        r.append(x.copy())
    assert_equal(r, [[2, 3], [1, 3], [1, 2]])


def test_resample_2d():
    data = ((1, 2), (3, 4), (5, 6))

    r = []
    for x in resample(data):
        r.append(x.copy())
    assert_equal(r, [[(3, 4), (5, 6)], [(1, 2), (5, 6)], [(1, 2), (3, 4)]])


def test_jackknife():
    data = (1, 2, 3)
    r = jackknife(lambda x: x.copy(), data)
    assert_equal(r, [[2, 3], [1, 3], [1, 2]])


def test_bias_on_unbiased():
    data = (0, 1, 2, 3)
    # bias is exactly zero for linear functions
    r = bias(np.mean, data)
    assert r == 0


def test_bias_on_biased_order_n_minus_one():
    # this "mean" has a bias of exactly O(n^{-1})
    def bad_mean(x):
        return (np.sum(x) + 2) / len(x)

    data = (0, 1, 2)
    r = bias(bad_mean, data)
    mean_jk = np.mean([bad_mean([1, 2]), bad_mean([0, 2]), bad_mean([0, 1])])
    # (5/2 + 4/2 + 3/2) / 3 = 12 / 6 = 2
    assert mean_jk == 2.0
    # f = 5/3
    # (n-1) * (mean_jk - f)
    # (3 - 1) * (6/3 - 5/3) = 2/3
    # note: 2/3 is exactly the bias of bad_mean for n = 3
    assert r == pytest.approx(2.0 / 3.0)


def test_bias_on_array_map():
    # compute mean and (biased) variance simultanously
    def fn(x):
        return np.mean(x), np.var(x, ddof=0)

    data = (0, 1, 2)
    r = bias(fn, data)
    assert_almost_equal(r, (0.0, -1.0 / 3.0))


def test_bias_corrected():
    # this "mean" has a bias of exactly O(n^{-1})
    def bad_mean(x):
        return (np.sum(x) + 2) / len(x)

    # bias correction is exact up to O(n^{-1})
    data = (0, 1, 2)
    r = bias_corrected(bad_mean, data)
    assert r == 1.0  # which is the correct unbiased mean


def test_variance():
    data = (0, 1, 2)
    r = variance(np.mean, data)
    # formula is (n - 1) / n * sum((jf - mean(jf)) ** 2)
    # fj = [3/2, 1, 1/2]
    # mfj = 1
    # ((3/2 - 1)^2 + (1 - 1)^2 + (1/2 - 1)^2) * 2 / 3
    # (1/4 + 1/4) / 3 * 2 = 1/3
    assert r == pytest.approx(1.0 / 3.0)
