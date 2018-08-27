import pytest
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


def test_jackknife_known_bias():
    est = jackknife_bias(x, np.mean)
    assert np.isclose(est, 0)


def test_jackknife_known_var():
    est = jackknife_variance(x, np.mean)
    assert np.isclose(est, np.var(x, ddof=1) / len(x))
