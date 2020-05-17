import numpy as np

from resample import bootstrap


class Jackknife:
    params = [100, 1000, 10_000]
    param_names = ["n"]

    def setup(self, n):
        self.arr = np.random.randn(n)

    def time_jackknife(self, n):
        bootstrap.jackknife(self.arr)

    def time_jackknife_bias(self, n):
        bootstrap.jackknife_bias(self.arr, f=np.mean)

    def time_jackknife_variance(self, n):
        bootstrap.jackknife_variance(self.arr, f=np.mean)


class Bootstrap:
    params = [[10_000], [100, 1000]]
    param_names = ["n", "b"]

    def setup(self, n, b):
        self.arr = np.random.randn(n)

    def time_bootstrap(self, n, b):
        bootstrap.bootstrap(self.arr, b=b)


class BootstrapCI:
    params = [[10_000], [1000], ["percentile", "bca", "t"]]
    param_names = ["n", "b", "ci_method"]

    def setup(self, n, b, ci_method):
        self.arr = np.random.randn(n)

    def time_bootstrap_ci(self, n, b, ci_method):
        bootstrap.bootstrap_ci(self.arr, f=np.mean, b=b, ci_method=ci_method)
