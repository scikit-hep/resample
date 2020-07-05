import pytest
import numpy as np
from resample.bootstrap import resample, confidence_interval
from numpy.testing import assert_equal


def run_resample(n, method):
    x = np.arange(n)
    r = []
    for _ in resample(x, method=method):
        r.append(x)
    return r


@pytest.mark.benchmark(group="bootstrap-100")
@pytest.mark.parametrize("method", ("ordinary", "balanced", "normal"))
def test_resample_100(benchmark, method):
    benchmark(run_resample, 100, method)


@pytest.mark.benchmark(group="bootstrap-1000")
@pytest.mark.parametrize("method", ("ordinary", "balanced", "normal"))
def test_bootstrap_resample_1000(benchmark, method):
    benchmark(run_resample, 1000, method)


@pytest.mark.benchmark(group="bootstrap-10000")
@pytest.mark.parametrize("method", ("ordinary", "balanced", "normal"))
def test_bootstrap_resample_10000(benchmark, method):
    benchmark(run_resample, 10000, method)


def run_confidence_interval(n, ci_method):
    x = np.arange(n)
    confidence_interval(np.mean, x, ci_method=ci_method)


@pytest.mark.benchmark(group="confidence-interval-100")
@pytest.mark.parametrize("ci_method", ("percentile", "student", "bca"))
def test_bootstrap_confidence_interval_100(benchmark, ci_method):
    benchmark(run_confidence_interval, 100, ci_method)


@pytest.mark.benchmark(group="confidence-interval-1000")
@pytest.mark.parametrize("ci_method", ("percentile", "student", "bca"))
def test_bootstrap_confidence_interval_1000(benchmark, ci_method):
    benchmark(run_confidence_interval, 1000, ci_method)


@pytest.mark.benchmark(group="confidence-interval-10000")
@pytest.mark.parametrize("ci_method", ("percentile", "student", "bca"))
def test_bootstrap_confidence_interval_10000(benchmark, ci_method):
    benchmark(run_confidence_interval, 10000, ci_method)
