import numpy as np
import pytest

from resample.bootstrap import (
    bootstrap,
    bootstrap_ci,
    empirical_influence,
    jackknife,
    jackknife_bias,
    jackknife_variance,
)
from resample.utils import ecdf, sup_norm

n = 100
b = 100
x = np.random.random(n)
f = ecdf(x)


def test_jackknife_shape():
    jack = jackknife(x)
    assert jack.shape == (n, n - 1)


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
    # TODO: Test error message
    with pytest.raises(ValueError):
        bootstrap(
            np.random.normal(size=(10, 2)), method="parametric", family="gaussian"
        )


def test_parametric_bootstrap_invalid_family_raises():
    # TODO: Test error message
    with pytest.raises(ValueError):
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
    # TODO: Test error message
    with pytest.raises(ValueError):
        bootstrap(x, strata=np.arange(len(x) + 1))


def test_bootstrap_invalid_method_raises():
    # TODO: Test error message
    with pytest.raises(ValueError):
        bootstrap(x, method="____")


def test_jackknife_known_bias():
    est = jackknife_bias(x, np.mean)
    assert np.isclose(est, 0)


def test_jackknife_known_variance():
    est = jackknife_variance(x, np.mean)
    assert np.isclose(est, np.var(x, ddof=1) / len(x))


def test_bootstrap_ci_invalid_p_raises():
    # TODO: Test error message
    with pytest.raises(ValueError):
        bootstrap_ci(x, f=np.mean, p=2)


@pytest.mark.parametrize("ci_method", ["percentile", "bca", "t"])
def test_bootstrap_ci_len(ci_method):
    ci = bootstrap_ci(x, f=np.mean, ci_method=ci_method)
    assert len(ci) == 2


def test_bootstrap_ci_invalid_boot_method_raises():
    # TODO: Test error message
    with pytest.raises(ValueError):
        bootstrap_ci(x, f=np.mean, boot_method="____")


def test_bootstrap_ci_invalid_ci_method_raises():
    # TODO: Test error message
    with pytest.raises(ValueError):
        bootstrap_ci(x, f=np.mean, ci_method="____")
