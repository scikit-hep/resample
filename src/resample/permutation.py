"""
Permutation-based equivalence tests
===================================

A collection of statistical tests of the null hypothesis that two or more samples are
drawn from the same population. The included tests check either whether the samples have
compatible means, medians, or use other sample properties. Permutations are used to
compute the distribution of the test statistic under the null hypothesis to obtain
accurate p-values without relying on approximate asymptotic formulas.

The permutation method is generic, it can be used with any test statistic, therefore we
also provide a generic test function that accepts a user-defined function to compute the
test statistic and then automatically computes the p-value for that statistic. The other
tests internally also call this generic test function.

All tests return a TestResult object, which mimics the interface of the result
objects returned by tests in scipy.stats, but has a third field to return the
estimated distribution of the test statistic under the null hypothesis.

Further reading:
https://en.wikipedia.org/wiki/P-value
https://en.wikipedia.org/wiki/Test_statistic
https://en.wikipedia.org/wiki/Paired_difference_test
"""
import sys
import typing as _tp
from dataclasses import dataclass as dataclass

import numpy as np
from numpy.typing import ArrayLike as _ArrayLike
from scipy.stats import rankdata as _rankdata
from scipy.stats import tiecorrect as _tiecorrect

from ._util import _normalize_rng
from .empirical import cdf_gen

_dataclass_kwargs = {"frozen": True, "repr": False}
if sys.version_info >= (3, 10):
    _dataclass_kwargs["slots"] = True  # pragma: no cover

_Kwargs = _tp.Any


@dataclass(**_dataclass_kwargs)
class TestResult:
    """
    Holder of the result of the permutation test.

    This class acts like a tuple, which means its can be unpacked and the fields can be
    accessed by name or by index.

    Attributes
    ----------
    statistic: float
        Value of the test statistic computed on the original data
    pvalue: float
        Chance probability (aka Type I error) for rejecting the null hypothesis. See
        https://en.wikipedia.org/wiki/P-value for details.
    samples: array
        Values of the test statistic from the permutated samples.
    """

    statistic: float
    pvalue: float
    samples: np.ndarray

    def __repr__(self) -> str:
        s = None
        if len(self.samples) < 7:
            s = str(self.samples)
        else:
            s = "[{0}, {1}, {2}, ..., {3}, {4}, {5}]".format(
                *self.samples[:3], *self.samples[-3:]
            )
        return "<TestResult statistic={0} pvalue={1} samples={2}>".format(
            self.statistic, self.pvalue, s
        )

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        if idx == 0:
            return self.statistic
        elif idx == 1:
            return self.pvalue
        elif idx == 2:
            return self.samples
        raise IndexError


def test(
    fn: _tp.Callable,
    x: _ArrayLike,
    y: _ArrayLike,
    *args: np.ndarray,
    transform: _tp.Optional[_tp.Callable] = None,
    size: int = 1000,
    random_state: _tp.Optional[_tp.Union[np.random.Generator, int]] = None,
) -> np.ndarray:
    """
    Compute p-value for null hypothesis that samples are drawn from the same population.

    We computes the p-value for the null hypothesis that the samples are drawn from the
    same population, based on the user-defined test statistic.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like
        Second sample.
    *args: array-like, optional
        Further samples.
    transform : Callable, optional
        Function f(x) for the test statistic to turn it into a measure of
        deviation. Must be vectorised.
    size : int, optional
        Number of permutations. Default 1000.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Returns
    -------
    TestResult
    """
    rng = _normalize_rng(random_state)
    r = []
    for arg in (x, y) + args:
        a = np.array(arg)
        if a.dtype.kind == "f" and np.any(np.isnan(a)):
            raise ValueError("input contains NaN")
        if len(a) < 2:
            raise ValueError("input arrays must have at least length 2")
        r.append(a)
    args = r
    del r

    # compute test statistic for original input
    t = fn(*args)

    # compute test statistic for permutated inputs
    arr = np.concatenate(args)
    slices = []
    start = 0
    for a in args:
        stop = start + len(a)
        slices.append(slice(start, stop))
        start = stop

    ts = np.empty(size)
    for i in range(size):
        rng.shuffle(arr)
        ts[i] = fn(*(arr[sl] for sl in slices))

    if transform is None:
        pvalue = np.mean(t < ts)
    else:
        pvalue = np.mean(transform(t) < transform(ts))
    return TestResult(t, pvalue, ts)


def anova(
    x: _ArrayLike, y: _ArrayLike, *args: _ArrayLike, **kwargs: _Kwargs
) -> TestResult:
    """
    Test whether the means of two or more samples are compatible.

    This test uses one-way analysis of variance (one-way ANOVA), see
    https://en.wikipedia.org/wiki/One-way_analysis_of_variance and
    https://en.wikipedia.org/wiki/F-test for details. This test is typically used when
    one has three groups or more. For two groups, Welch's ttest is preferred, because
    ANOVA assumes equal variances for the samples.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like
        Second sample.
    *args : array-like
        Further samples.
    **kwargs :
        Keyword arguments are forward to :meth:`test`.

    Returns
    -------
    TestResult
    """
    del kwargs["transform"]
    return test(_ANOVA(), x, y, *args, **kwargs)


def mannwhitneyu(x: _ArrayLike, y: _ArrayLike, **kwargs: _Kwargs) -> TestResult:
    """
    Test whether two samples are drawn from the same population based on ranking.

    This performs the permutation-based Mann-Whitney U test, see
    https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test for details. The test
    works for any population of samples that are ordinal, so also for integers. This is
    the two-sided version of the test, meaning that the test gives the same pvalue if
    x and y are swapped.

    For normally distributed data, the test is almost as powerful as the t-test and
    considerably more powerful for non-normal populations. It should be kept in mind,
    however, that the two tests are not equivalent. The t-test tests whether
    two samples have the same mean and ignores differences in variance, while the
    Mann-Whitney U test would detect differences in variance.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like
        Second sample.
    **kwargs :
        Keyword arguments are forward to :meth:`test`.
    **kwargs :
        Keyword arguments are forward to :meth:`test`.

    Returns
    -------
    TestResult
    """
    n1 = len(x)
    n2 = len(y)
    mu = n1 * n2 // 2
    del kwargs["transform"]
    return test(_mannwhitneyu, x, y, transform=lambda x: np.abs(x - mu), **kwargs)


def kruskal(
    x: _ArrayLike, y: _ArrayLike, *args: _ArrayLike, **kwargs: _Kwargs
) -> TestResult:
    """
    Test whether two or more samples are drawn from the same population.

    This performs a permutation-based Kruskal-Wallis test, see
    https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance
    for details. It extends the Mann-Whitney U test to more than two groups.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like
        Second sample.
    *args : array-like
        Further samples.
    **kwargs :
        Keyword arguments are forward to :meth:`test`.

    Returns
    -------
    TestResult
    """
    del kwargs["transform"]
    return test(_kruskal, x, y, *args, **kwargs)


def ks(x: _ArrayLike, y: _ArrayLike, **kwargs: _Kwargs) -> TestResult:
    """
    Test whether two samples are drawn from the same population.

    This performs the permutation-based two-sided Kolmogorov-Smirnov test.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like
        Second sample.
    **kwargs :
        Keyword arguments are forward to :meth:`test`.

    Returns
    -------
    TestResult
    """
    return test(_KS(), x, y, **kwargs)


def pearson(x: _ArrayLike, y: _ArrayLike, **kwargs: _Kwargs) -> TestResult:
    """
    Perform permutation-based correlation test.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like
        Second sample.
    **kwargs :
        Keyword arguments are forward to :meth:`test`.

    Returns
    -------
    TestResult
    """
    if len(x) != len(y):
        raise ValueError("x and y must have have the same length")
    if len(x) < 2:
        raise ValueError("length of x and y must be at least 2.")
    del kwargs["transform"]
    return test(_pearson, x, y, transform=np.abs, **kwargs)


def spearman(x: _ArrayLike, y: _ArrayLike, **kwargs: _Kwargs) -> TestResult:
    """
    Perform permutation-based correlation test of rank data.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like
        Second sample.
    **kwargs :
        Keyword arguments are forward to :meth:`test`.

    Returns
    -------
    TestResult
    """
    if len(x) != len(y):
        raise ValueError("x and y must have have the same length")
    if len(x) < 2:
        raise ValueError("length of x and y must be at least 2.")
    del kwargs["transform"]
    return test(_spearman, x, y, transform=np.abs, **kwargs)


def ttest(x: _ArrayLike, y: _ArrayLike, **kwargs: _Kwargs) -> TestResult:
    """
    Test whether the means of two samples are compatible with Welch's t-test.

    See https://en.wikipedia.org/wiki/Welch%27s_t-test for details on this test. The
    p-value computed is for the null hypothesis that the two population means are equal.
    The test is two-sided, which means that swapping x and y gives the same pvalue.
    Welch's t-test does not require the sample sizes to be equal and it does not require
    the samples to have the same variance.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like
        Second sample.
    **kwargs :
        Keyword arguments are forward to :meth:`test`.

    Returns
    -------
    TestResult
    """
    del kwargs["transform"]
    return test(_ttest, x, y, transform=np.abs, **kwargs)


def usp(w: _ArrayLike, **kwargs: _Kwargs):
    """
    Test independence of two data sets based on a two-dimensional histogram.

    To apply this test, generate a two-dimensional histogram from the pairs of input
    data (X_i, Y_i). Pass the resulting 2D histogram to this function. The histogram
    does not have to be square.

    The test is the U-statistic permutation (USP) test of independence for
    discrete data displayed in a contigency table, paper:
    https://doi.org/10.1098/rspa.2021.0549
    According to the papeer, it outperforms the Pearson's chi^2 and the G-test, both
    in stability and power.

    Parameters
    ----------
    w : array-like
        2D histgram of (X, Y) pairs.
    **kwargs :
        Keyword arguments are forward to :meth:`test`.

    Returns
    -------
    TestResult
    """
    w = np.atleast_2d(w)
    if w.ndim != 2:
        raise ValueError("w must be a 2D array-like")
    if w.shape[0] < 2 or w.shape[1] < 2:
        raise ValueError("w must have at least two entries per dimension")
    return test(_usp, w, **kwargs)


def _usp(w: np.ndarray) -> float:
    return 0.0


def _ttest(x: np.ndarray, y: np.ndarray) -> float:
    n1 = len(x)
    n2 = len(y)
    m1 = np.mean(x)
    m2 = np.mean(y)
    v1 = np.var(x, ddof=1)
    v2 = np.var(y, ddof=1)
    r: float = (m1 - m2) / np.sqrt(v1 / n1 + v2 / n2)
    return r


def _mannwhitneyu(x: np.ndarray, y: np.ndarray) -> float:
    # method 2 from Wikipedia, but returning U1 instead of min(U1, U2) to be
    # consistent with scipy.stats.mannwhitneyu(x, y, alternative="two-sided")
    n1 = len(x)
    a = _rankdata(np.concatenate([x, y]))
    r1 = np.sum(a[:n1])
    u1: float = r1 - 0.5 * n1 * (n1 + 1)
    return u1


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    m1 = np.mean(x)
    m2 = np.mean(y)
    s1 = np.mean((x - m1) ** 2)
    s2 = np.mean((y - m2) ** 2)
    r: float = np.mean((x - m1) * (y - m2)) / np.sqrt(s1 * s2)
    return r


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = _rankdata(x)
    y = _rankdata(y)
    return _pearson(x, y)


def _kruskal(*args: np.ndarray) -> float:
    # see https://en.wikipedia.org/wiki/
    #           Kruskal%E2%80%93Wallis_one-way_analysis_of_variance
    # method 3 and 4
    joined = np.concatenate(args)
    r = _rankdata(joined)
    n = len(r)
    start = 0
    r_args = []
    for i, a in enumerate(args):
        r_args.append(r[start : start + len(a)])
        start += len(a)

    # method 3 (assuming no ties)
    h = 12.0 / (n * (n + 1)) * sum(len(r) * np.mean(r) ** 2 for r in r_args) - 3 * (
        n + 1
    )

    # apply tie correction
    h /= _tiecorrect(r)
    return h


class _ANOVA:
    # see https://en.wikipedia.org/wiki/F-test
    km1: int = -2
    nmk: int = 0
    a_bar: float = 0.0

    def __call__(self, *args: np.ndarray) -> float:
        if self.km1 == -2:
            self._init(args)

        between_group_variability = (
            sum(len(a) * (np.mean(a) - self.a_bar) ** 2 for a in args) / self.km1
        )
        within_group_variability = sum(len(a) * np.var(a) for a in args) / (self.nmk)
        return between_group_variability / within_group_variability

    def _init(self, args: _tp.Tuple[np.ndarray, ...]) -> None:
        n = sum(len(a) for a in args)
        k = len(args)
        self.km1 = k - 1
        self.nmk = n - k
        self.a_bar = np.mean(np.concatenate(args))


class _KS:
    all = None

    def __call__(self, *args: np.ndarray) -> float:
        if self.all is None:
            self._init(args)
        x, y = args
        f1 = cdf_gen(x)
        f2 = cdf_gen(y)
        d = f1(self.all) - f2(self.all)
        d1 = np.clip(-np.min(d), 0, 1)
        d2 = np.max(d)
        return max(d1, d2)

    def _init(self, args: _tp.Tuple[np.ndarray, ...]) -> None:
        self.all = np.concatenate(args)
