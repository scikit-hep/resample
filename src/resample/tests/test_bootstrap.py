import numpy as np
import pytest
from numpy.testing import assert_equal

from resample.bootstrap import (
    bootstrap,
    bootstrap_ci,
    empirical_influence,
    jackknife,
    jackknife_bias,
    jackknife_bias_corrected,
    jackknife_variance,
)
from resample.utils import ecdf, sup_norm

n = 100
b = 100
x = np.random.random(n)
f = ecdf(x)


def test_jackknife():
    x = [0, 1, 2, 3]
    r = jackknife(x, lambda x: x.copy())
    assert_equal(r, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])


def test_jackknife_bias_unbiased():
    x = [0, 1, 2, 3]
    # bias is exactly zero for linear functions
    r = jackknife_bias(x, np.mean)
    assert r == 0


def test_jackknife_bias_order_n_minus_one():
    # this "mean" has a bias of exactly O(n^{-1})
    def bad_mean(x):
        return (np.sum(x) + 2) / len(x)

    x = [0, 1, 2]
    r = jackknife_bias(x, bad_mean)
    mean_jk = np.mean([bad_mean([1, 2]), bad_mean([0, 2]), bad_mean([0, 1])])
    # (5/2 + 4/2 + 3/2) / 3 = 12 / 6 = 2
    assert mean_jk == 2.0
    # f = 5/3
    # (n-1) * (mean_jk - f)
    # (3 - 1) * (6/3 - 5/3) = 2/3
    # note: 2/3 is exactly the bias of bad_mean for n = 3
    assert r == pytest.approx(2.0 / 3.0)


def test_jackknife_bias_corrected():
    # this "mean" has a bias of exactly O(n^{-1})
    def bad_mean(x):
        return (np.sum(x) + 2) / len(x)

    # bias correction is exact up to O(n^{-1})
    x = [0, 1, 2]
    r = jackknife_bias_corrected(x, bad_mean)
    assert r == 1.0  # which is the correct unbiased mean


def test_jackknife_variance():
    x = [0, 1, 2]
    r = jackknife_variance(x, np.mean)
    # formula is (n - 1) / n * sum((jf - mean(jf)) ** 2)
    # fj = [3/2, 1, 1/2]
    # mfj = 1
    # ((3/2 - 1)^2 + (1 - 1)^2 + (1/2 - 1)^2) * 2 / 3
    # (1/4 + 1/4) / 3 * 2 = 1/3
    assert r == pytest.approx(1.0 / 3.0)


def test_empirical_influence_shape():
    emp = empirical_influence(x, f=np.mean)
    assert len(emp) == n


def test_ordinary_bootstrap_shape():
    boot = bootstrap(x, b=b, method="ordinary")
    assert boot.shape == (b, n)


def test_balanced_bootstrap_distributions_equal():
    xbal = np.ravel(bootstrap(x, method="balanced"))
    g = ecdf(xbal)
    assert sup_norm(f, g, (-10, 10)) == 0.0


def test_parametric_bootstrap_multivariate_raises():
    msg = "must be one-dimensional"
    with pytest.raises(ValueError, match=msg):
        bootstrap(
            np.random.normal(size=(10, 2)), method="parametric", family="gaussian"
        )


def test_parametric_bootstrap_invalid_family_raises():
    msg = "Invalid family"
    with pytest.raises(ValueError, match=msg):
        bootstrap(x, method="parametric", family="____")


@pytest.mark.parametrize(
    "family",
    [
        "gaussian",
        "t",
        "laplace",
        "logistic",
        "F",
        "gamma",
        "log-normal",
        "inverse-gaussian",
        "pareto",
        "beta",
        "poisson",
    ],
)
def test_parametric_bootstrap_shape(family):
    boot = bootstrap(x, b=b, method="parametric", family=family)
    assert boot.shape == (b, n)


def test_bootstrap_shape():
    arr = np.random.normal(size=(16, 8, 4, 2))
    boot = bootstrap(arr, b=b)
    assert boot.shape == (b, 16, 8, 4, 2)


def test_bootstrap_equal_along_axis():
    arr = np.reshape(np.tile([0, 1, 2], 3), newshape=(3, 3))
    boot = bootstrap(arr, b=10)
    assert np.all([np.array_equal(arr, a) for a in boot])


def test_bootstrap_full_strata():
    boot = bootstrap(x, b=b, strata=np.array(range(n)))
    assert np.all([np.array_equal(x, a) for a in boot])


def test_bootstrap_invalid_strata_raises():
    msg = "must have the same length"
    with pytest.raises(ValueError, match=msg):
        bootstrap(x, strata=np.arange(len(x) + 1))


def test_bootstrap_invalid_method_raises():
    msg = "method must be either 'ordinary', 'balanced', or 'parametric'"
    with pytest.raises(ValueError, match=msg):
        bootstrap(x, method="____")


def test_bootstrap_ci_invalid_p_raises():
    msg = "p must be between zero and one"
    with pytest.raises(ValueError, match=msg):
        bootstrap_ci(x, f=np.mean, p=2)


@pytest.mark.parametrize("ci_method", ["percentile", "bca", "t"])
def test_bootstrap_ci_len(ci_method):
    ci = bootstrap_ci(x, f=np.mean, ci_method=ci_method)
    assert len(ci) == 2


def test_bootstrap_ci_invalid_boot_method_raises():
    msg = "must be 'ordinary', 'balanced', or 'parametric'"
    with pytest.raises(ValueError, match=msg):
        bootstrap_ci(x, f=np.mean, boot_method="____")


def test_bootstrap_ci_invalid_ci_method_raises():
    msg = "method must be 'percentile', 'bca', or 't'"
    with pytest.raises(ValueError, match=msg):
        bootstrap_ci(x, f=np.mean, ci_method="____")
