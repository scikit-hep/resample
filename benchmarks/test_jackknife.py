# ruff: noqa: D100 D103
import numpy as np
import pytest
from numpy.testing import assert_equal
from resample.jackknife import resample


def run_resample(n, copy):
    x = np.arange(n)
    r = []
    for b in resample(x, copy=copy):
        r.append(np.mean(b))
    return r


@pytest.mark.benchmark(group="jackknife-100")
@pytest.mark.parametrize("copy", (True, False))
def test_jackknife_resample_100(benchmark, copy):
    result = benchmark(run_resample, 100, copy)
    assert_equal(result, run_resample(100, resample))


@pytest.mark.benchmark(group="jackknife-1000")
@pytest.mark.parametrize("copy", (True, False))
def test_jackknife_resample_1000(benchmark, copy):
    result = benchmark(run_resample, 1000, copy)
    assert_equal(result, run_resample(1000, resample))
