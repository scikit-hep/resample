import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest

from resample.jackknife import bias, bias_corrected, jackknife, variance


def test_jackknife():
    x = [0, 1, 2, 3]
    r = jackknife(x, lambda x: x.copy())
    assert_equal(r, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


def test_jackknife_bias_unbiased():
    x = [0, 1, 2, 3]
    # bias is exactly zero for linear functions
    r = bias(x, np.mean)
    assert r == 0


def test_jackknife_bias_order_n_minus_one():
    # this "mean" has a bias of exactly O(n^{-1})
    def bad_mean(x):
        return (np.sum(x) + 2) / len(x)

    x = [0, 1, 2]
    r = bias(x, bad_mean)
    mean_jk = np.mean([bad_mean([1, 2]), bad_mean([0, 2]), bad_mean([0, 1])])
    # (5/2 + 4/2 + 3/2) / 3 = 12 / 6 = 2
    assert mean_jk == 2.0
    # f = 5/3
    # (n-1) * (mean_jk - f)
    # (3 - 1) * (6/3 - 5/3) = 2/3
    # note: 2/3 is exactly the bias of bad_mean for n = 3
    assert r == pytest.approx(2.0 / 3.0)


def test_jackknife_bias_array_map():
    # compute mean and (biased) variance simultanously
    def fcn(x):
        return np.mean(x), np.var(x, ddof=0)

    x = [0, 1, 2]
    r = bias(x, fcn)
    assert_almost_equal(r, (0.0, -1.0 / 3.0))


def test_jackknife_bias_corrected():
    # this "mean" has a bias of exactly O(n^{-1})
    def bad_mean(x):
        return (np.sum(x) + 2) / len(x)

    # bias correction is exact up to O(n^{-1})
    x = [0, 1, 2]
    r = bias_corrected(x, bad_mean)
    assert r == 1.0  # which is the correct unbiased mean


def test_jackknife_variance():
    x = [0, 1, 2]
    r = variance(x, np.mean)
    # formula is (n - 1) / n * sum((jf - mean(jf)) ** 2)
    # fj = [3/2, 1, 1/2]
    # mfj = 1
    # ((3/2 - 1)^2 + (1 - 1)^2 + (1/2 - 1)^2) * 2 / 3
    # (1/4 + 1/4) / 3 * 2 = 1/3
    assert r == pytest.approx(1.0 / 3.0)
