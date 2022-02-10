"""
Permutation-based tests.

A collection of statistical tests that use permutated samples. Permutations are used to
compute the distribution of a test statistic under some null hypothesis to obtain
p-values without relying on approximate asymptotic formulas.

The permutation method is generic, it can be used with any test statistic, therefore we
also provide a generic test function that accepts a user-defined function to compute the
test statistic and then automatically computes the p-value for that statistic. The other
tests internally also call this generic test function.

All tests return a TestResult object, which mimics the interface of the result
objects returned by tests in scipy.stats, but has a third field to return the
estimated distribution of the test statistic under the null hypothesis.

Further reading:

- https://en.wikipedia.org/wiki/P-value
- https://en.wikipedia.org/wiki/Test_statistic
- https://en.wikipedia.org/wiki/Paired_difference_test
"""
import sys
import typing as _tp
from dataclasses import dataclass as dataclass

import numpy as np
from scipy.stats import rankdata as _rankdata
from scipy.stats import tiecorrect as _tiecorrect

from ._util import _fill_w, _normalize_rng

_dataclass_kwargs = {"frozen": True, "repr": False}
if sys.version_info >= (3, 10):
    _dataclass_kwargs["slots"] = True  # pragma: no cover

_Kwargs = _tp.Any
_ArrayLike = _tp.Collection


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
        Estimated chance probability (aka Type I error) for rejecting the null
        hypothesis. See https://en.wikipedia.org/wiki/P-value for details.
    interval: (float, float)
        Standard interval (approximately 68 % coverage) for the p-value. This interval
        reflects the statistical uncertainty from doing only a finite number of random
        permutations instead of infinitely many. The interval can be narrowed down by
        running the test with more permutations. This interval does not have exact
        coverage for finite samples.
    samples: array
        Values of the test statistic from the permutated samples.
    """

    statistic: float
    pvalue: float
    interval: _tp.Tuple[float, float]
    samples: np.ndarray

    def __repr__(self) -> str:
        """Return (potentially shortened) representation."""
        s = None
        if len(self.samples) < 7:
            s = str(self.samples)
        else:
            s = "[{0}, {1}, {2}, ..., {3}, {4}, {5}]".format(
                *self.samples[:3], *self.samples[-3:]
            )
        return "<TestResult statistic={0} pvalue={1} interval={2} samples={3}>".format(
            self.statistic, self.pvalue, self.interval, s
        )

    def __len__(self):
        """Return length of tuple."""
        return 4

    def __getitem__(self, idx):
        """Return fields by index."""
        if idx == 0:
            return self.statistic
        elif idx == 1:
            return self.pvalue
        elif idx == 2:
            return self.interval
        elif idx == 3:
            return self.samples
        raise IndexError


def usp(
    w: _ArrayLike,
    *,
    precision: float = 0.01,
    max_size: int = 10000,
    random_state: _tp.Optional[_tp.Union[np.random.Generator, int]] = None,
):
    """
    Test independence of two discrete data sets with the U-statistic.

    The USP test is described in this paper: https://doi.org/10.1098/rspa.2021.0549.
    According to the paper, it outperforms the Pearson's χ² and the G-test in both
    in stability and power.

    It requires that w is a 2d histogram of the value pairs. Whether the original
    values were discrete or continuous does not matter for the test. Using a large
    number bins is safe, since the test is not negatively affected by bins with
    zero entries.

    Parameters
    ----------
    w : array-like
        Two-dimensional array which represents the counts in a histogram. The counts
        can be of floating point type, but must have integral values.
    precision : float, optional
        Target precision (statistical) for the p-value. The algorithm estimates the
        number of required permutations to reach the target precision. The accuracy of
        the actual value may be above or below this value. If precision is zero,
        max_size permutations are used. Default 0.01.
    max_size : int, optional
        Maximum number of permutations. Default 10000.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Returns
    -------
    TestResult
    """
    if precision < 0:
        raise ValueError("precision cannot be negative")

    if max_size <= 0:
        raise ValueError("max_size must be positive")

    rng = _normalize_rng(random_state)

    w = np.asarray(w, dtype=int)
    if w.ndim != 2:
        raise ValueError("w must be two-dimensional")
    wx = np.sum(w, axis=1)
    wy = np.sum(w, axis=0)
    n = np.sum(wx)

    m = np.outer(wx, wy).astype(float) / n

    f1 = 1.0 / (n * (n - 3))
    f2 = 4.0 / (n * (n - 2) * (n - 3))

    t = _usp(f1, f2, w, m)

    # generate x,y index arrays
    xmap = np.empty(n, dtype=int)
    ymap = np.empty(n, dtype=int)
    k = 0
    for ix in range(w.shape[0]):
        for iy in range(w.shape[1]):
            wij = int(w[ix, iy])
            xmap[k : k + wij] = ix
            ymap[k : k + wij] = iy
            k += wij

    # For Type I error probabilities to hold theoretically, the number of bootstrap
    # samples drawn may not the depend on the data (comment by Richard Samworth).
    # However, computing the required number of samples with the worst-case p=0.5
    # can lead to very pessimistic estimates for the required number of samples,
    # up to orders of magnitude worse. As a compromise, we burn 10 samples to compute
    # a crude estimate of the p-value and then use that to draw a fresh sample to
    # compute the final p-value.
    niter = 2 if precision > 0 else 1
    n = 10 if niter > 1 else max_size
    for iter in range(niter):
        ts = np.empty(n)
        for b in range(n):
            rng.shuffle(ymap)
            _fill_w(w, xmap, ymap)
            # m stays the same, since wx and wy remain unchanged
            ts[b] = _usp(f1, f2, w, m)
        if iter == 0 and niter > 1:
            p = _wilson_center(np.mean(t < ts), n)
            n = min(int(p * (1 - p) / precision**2), max_size)
        else:
            pvalue, interval = _wilson_score_interval(np.sum(t < ts), n, 1.0)

    return TestResult(t, pvalue, interval, ts)


def _usp(f1, f2, w, m):
    # Eq. 2.1 from https://doi.org/10.1098/rspa.2021.0549
    return f1 * np.sum((w - m) ** 2) - f2 * np.sum(w * m)


def same_population(
    fn: _tp.Callable,
    x: _ArrayLike,
    y: _ArrayLike,
    *args: _ArrayLike,
    transform: _tp.Optional[_tp.Callable] = None,
    precision: float = 0.01,
    max_size: int = 10000,
    random_state: _tp.Optional[_tp.Union[np.random.Generator, int]] = None,
) -> np.ndarray:
    """
    Compute p-value for null hypothesis that samples are drawn from the same population.

    The computation is based on a user-defined test statistic. The distribution of the
    test statistic under the null hypothesis is generated by generating random
    permutations of the inputs, to simulate that they are actually drawn from the same
    population. The test statistic is recomputed on these permutations and the p-value
    is computed as the fraction of these resampled test statistics which are larger
    than the original value.

    Some test statistics need to be transformed to fulfill the condition above, for
    example if they are signed. A transform can be passed to this function for those
    cases.

    Parameters
    ----------
    fn : Callable
        Function with signature f(x, ...), where the number of arguments corresponds to
        the number of data samples passed to the test.
    x : array-like
        First sample.
    y : array-like
        Second sample.
    *args: array-like
        Further samples, if the test allows to compare more than two.
    transform : Callable, optional
        Function with signature f(x) for the test statistic to turn it into a measure of
        deviation. Must be vectorised.
    precision : float, optional
        Target precision (statistical) for the p-value. The algorithm estimates the
        number of required permutations to reach the target precision. The accuracy of
        the actual value may be above or below this value. If precision is zero,
        max_size permutations are used. Default 0.01.
    max_size : int, optional
        Maximum number of permutations. Default 10000.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Returns
    -------
    TestResult
    """
    if precision < 0:
        raise ValueError("precision cannot be negative")

    if max_size <= 0:
        raise ValueError("max_size must be positive")

    rng = _normalize_rng(random_state)

    r = []
    for arg in (x, y) + args:
        a = np.array(arg)
        if a.ndim != 1:
            raise ValueError("input samples must be 1D arrays")
        if len(a) < 2:
            raise ValueError("input arrays must have at least two items")
        if a.dtype.kind == "f" and np.any(np.isnan(a)):
            raise ValueError("input contains NaN")
        r.append(a)
    args = r
    del r

    # compute test statistic for original input
    t = fn(*args)

    # compute test statistic for permutated inputs
    slices = []
    start = 0
    for a in args:
        stop = start + len(a)
        slices.append(slice(start, stop))
        start = stop

    joined_sample = np.concatenate(args)

    # For algorithm below, see comment in usp function.
    niter = 2 if precision > 0 else 1
    n = 10 if niter > 1 else max_size
    for iter in range(niter):
        ts = np.empty(n)
        for b in range(n):
            rng.shuffle(joined_sample)
            ts[b] = fn(*(joined_sample[sl] for sl in slices))
        if transform is None:
            u = t
            us = ts
        else:
            u = transform(t)
            us = transform(ts)
        if iter == 0 and niter > 1:
            p = _wilson_center(np.mean(u < us), len(us))
            n = min(int(p * (1 - p) / precision**2), max_size)
        else:
            pvalue, interval = _wilson_score_interval(np.sum(u < us), n, 1.0)

    return TestResult(t, pvalue, interval, ts)


def anova(
    x: _ArrayLike, y: _ArrayLike, *args: _ArrayLike, **kwargs: _Kwargs
) -> TestResult:
    """
    Test whether the means of two or more samples are compatible.

    This test uses one-way analysis of variance (one-way ANOVA) which tests whether the
    samples have the same mean. This test is typically used when one has three groups
    or more. For two groups, Welch's ttest is preferred, because ANOVA assumes equal
    variances for the samples.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like
        Second sample.
    *args : array-like
        Further samples.
    **kwargs :
        Keyword arguments are forward to :meth:`same_population`.

    Returns
    -------
    TestResult

    Notes
    -----
    https://en.wikipedia.org/wiki/One-way_analysis_of_variance
    https://en.wikipedia.org/wiki/F-test
    """
    kwargs["transform"] = None
    return same_population(_ANOVA(), x, y, *args, **kwargs)


def kruskal(
    x: _ArrayLike, y: _ArrayLike, *args: _ArrayLike, **kwargs: _Kwargs
) -> TestResult:
    """
    Test whether two or more samples have the same mean rank.

    This performs a permutation-based Kruskal-Wallis test. In a sense, it extends the
    Mann-Whitney U test, which also uses ranks, to more than two groups. It does so by
    comparing the means of the rank distributions.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like
        Second sample.
    *args : array-like
        Further samples.
    **kwargs :
        Keyword arguments are forward to :meth:`same_population`.

    Returns
    -------
    TestResult

    Notes
    -----
    https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance
    """
    kwargs["transform"] = None
    return same_population(_kruskal, x, y, *args, **kwargs)


def pearsonr(x: _ArrayLike, y: _ArrayLike, **kwargs: _Kwargs) -> TestResult:
    """
    Test whether two samples are drawn from the same population using the correlation.

    The test statistic is the Pearson correlation coefficient. The test is very
    sensitive to linear relationship of x and y. If the relationship is very non-linear
    but monotonic, :func:`spearmanr` may be more sensitive.

    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like
        Second sample.
    **kwargs :
        Keyword arguments are forward to :meth:`same_population`.

    Returns
    -------
    TestResult
    """
    if len(x) != len(y):
        raise ValueError("x and y must have have the same length")
    kwargs["transform"] = np.abs
    return same_population(_pearson, x, y, **kwargs)


def spearmanr(x: _ArrayLike, y: _ArrayLike, **kwargs: _Kwargs) -> TestResult:
    """
    Test whether two samples are drawn from the same population using rank correlation.

    The test statistic is Spearman's rank correlation coefficient. The test is very
    sensitive to monotonic relationships between x and y, even if it is very non-linear.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like
        Second sample.
    **kwargs :
        Keyword arguments are forward to :meth:`same_population`.

    Returns
    -------
    TestResult
    """
    if len(x) != len(y):
        raise ValueError("x and y must have have the same length")
    kwargs["transform"] = np.abs
    return same_population(_spearman, x, y, **kwargs)


def ttest(x: _ArrayLike, y: _ArrayLike, **kwargs: _Kwargs) -> TestResult:
    """
    Test whether the means of two samples are compatible with Welch's t-test.

    See https://en.wikipedia.org/wiki/Welch%27s_t-test for details on this test. The
    p-value computed is for the null hypothesis that the two population means are equal.
    The test is two-sided, which means that swapping x and y gives the same p-value.
    Welch's t-test does not require the sample sizes to be equal and it does not require
    the samples to have the same variance.

    Parameters
    ----------
    x : array-like
        First sample.
    y : array-like
        Second sample.
    **kwargs :
        Keyword arguments are forward to :meth:`same_population`.

    Returns
    -------
    TestResult
    """
    kwargs["transform"] = np.abs
    return same_population(_ttest, x, y, **kwargs)


def _ttest(x: np.ndarray, y: np.ndarray) -> float:
    n1 = len(x)
    n2 = len(y)
    m1 = np.mean(x)
    m2 = np.mean(y)
    v1 = np.var(x, ddof=1)
    v2 = np.var(y, ddof=1)
    r: float = (m1 - m2) / np.sqrt(v1 / n1 + v2 / n2)
    return r


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


def _wilson_score_interval(n1, n, z):
    p = n1 / n
    norm = 1 / (1 + z**2 / n)
    a = p + 0.5 * z**2 / n
    b = z * np.sqrt(p * (1 - p) / n + 0.25 * (z / n) ** 2)
    return p, ((a - b) * norm, (a + b) * norm)


def _wilson_center(p, n):
    return (p + 0.5 / n) / (1.0 + 1.0 / n)
