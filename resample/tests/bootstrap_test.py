import numpy as np
from resample.utils import ecdf, sup_norm
from resample.bootstrap import (bootstrap,
                                jackknife_bias,
                                jackknife_variance)

np.random.seed(2357)

x = np.random.randn(100)
f = ecdf(x)


def test_balanced_bootstrap_eq_orig():
    xbal = np.ravel(bootstrap(x, method="balanced"))
    g = ecdf(xbal)
    assert sup_norm(f, g, (-10, 10)) == 0.0


def test_bootstrap_dim():
    arr = np.random.normal(size=(16, 8, 4, 2))
    boot = bootstrap(arr, b=100)
    assert boot.shape == (100, 16, 8, 4, 2)


def test_bootstrap_eq_along_axis():
    arr = np.reshape(np.tile([0, 1, 2], 3), newshape=(3, 3))
    boot = bootstrap(arr, b=10)
    assert np.all([np.array_equal(arr, a) for a in boot])


def test_jackknife_known_bias():
    est = jackknife_bias(x, np.mean)
    assert np.isclose(est, 0)


def test_jackknife_known_var():
    est = jackknife_variance(x, np.mean)
    assert np.isclose(est, np.var(x, ddof=1) / len(x))
