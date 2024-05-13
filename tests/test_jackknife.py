import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from scipy.optimize import curve_fit
from resample.jackknife import (
    bias,
    bias_corrected,
    jackknife,
    resample,
    variance,
    cross_validation,
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


def test_resample_several_args():
    a = [1, 2, 3]
    b = [(1, 2), (2, 3), (3, 4)]
    c = ["12", "3", "4"]
    for ai, bi, ci in resample(a, b, c):
        assert np.shape(ai) == (2,)
        assert np.shape(bi) == (2, 2)
        assert np.shape(ci) == (2,)
        assert set(ai) <= set(a)
        assert set(ci) <= set(c)
        bi = list(tuple(x) for x in bi)
        assert set(bi) <= set(b)


def test_resample_several_args_incompatible_keywords():
    a = [1, 2, 3]
    with pytest.raises(ValueError):
        resample(a, [1, 2])

    with pytest.raises(ValueError):
        resample(a, [1, 2, 3, 4])


def test_resample_deprecation():
    data = [1, 2, 3]

    with pytest.warns(FutureWarning):
        r = list(resample(data, False))

    assert_equal(r, list(resample(data, copy=False)))

    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError):  # too many arguments
            resample(data, True, 1)


@pytest.mark.filterwarnings("ignore:Covariance")
def test_cross_validation():
    x = [1, 2, 3]
    y = [3, 4, 5]

    def predict(xi, yi, xo, npar):
        def model(x, *par):
            return np.polyval(par, x)

        popt = curve_fit(model, xi, yi, p0=np.zeros(npar))[0]
        return model(xo, *popt)

    v = cross_validation(predict, x, y, 2)
    assert v == pytest.approx(0)

    v2 = cross_validation(predict, x, y, 1)
    assert v2 == pytest.approx(1.5)
