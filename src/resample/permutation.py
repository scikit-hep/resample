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

__all__ = [
    "TestResult",
    "usp",
    "same_population",
    "anova",
    "kruskal",
    "pearsonr",
    "spearmanr",
    "ttest",
]

import sys
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats as _stats

from . import _util

_dataclass_kwargs = {"frozen": True, "repr": False}
if sys.version_info >= (3, 10):
    _dataclass_kwargs["slots"] = True  # pragma: no cover


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
    samples: array
        Values of the test statistic from the permutated samples.

    """

    statistic: float
    pvalue: float
    samples: NDArray

    def __repr__(self) -> str:
        """Return (potentially shortened) representation."""
        s = None
        if len(self.samples) < 7:
            s = str(self.samples)
        else:
            s = "[{}, {}, {}, ..., {}, {}, {}]".format(
                *self.samples[:3], *self.samples[-3:]
            )
        return (
            f"<TestResult statistic={self.statistic} pvalue={self.pvalue} samples={s}>"
        )

    def __len__(self) -> int:
        """Return length of tuple."""
        return 3

    def __getitem__(self, idx: int) -> Union[float, NDArray]:
        """Return fields by index."""
        if idx == 0:
            return self.statistic
        elif idx == 1:
            return self.pvalue
        elif idx == 2:
            return self.samples
        raise IndexError


def usp(
    w: "ArrayLike",
    *,
    size: int = 9999,
    method: str = "auto",
    random_state: Optional[Union[np.random.Generator, int]] = None,
) -> TestResult:
    """
    Test independence of two discrete data sets with the U-statistic.

    The USP test is described in this paper: https://doi.org/10.1098/rspa.2021.0549.
    According to the paper, it outperforms the Pearson's χ² and the G-test in both in
    stability and power.

    It requires that the input is a contigency table (a 2D histogram of value pairs).
    Whether the original values were discrete or continuous does not matter for the
    test. In case of continuous values, using a large number of bins is safe, since the
    test is not negatively affected by bins with zero entries.

    Parameters
    ----------
    w : array-like
        Two-dimensional array which represents the counts in a histogram. The counts
        can be of floating point type, but must have integral values.
    size : int, optional
        Number of permutations. Default 9999.
    method : str, optional
        Method used to generate random tables under the null hypothesis.
        'auto': Use heuristic to select fastest algorithm for given table.
        'boyett': Boyett's algorithm, which requires extra space to store N + 1
        integers for N entries in total and has O(N) time complexity. It performs
        poorly when N is large, but does not depend on the number of K table cells.
        'patefield': Patefield's algorithm, which does not require extra space and
        has O(K log(N)) time complexity. It performs well even if N is huge. For
        small N and large K, the shuffling algorithm is faster.
        Default is 'auto'.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use ``numpy.random.default_rng()``.

    Returns
    -------
    TestResult

    """
    if size <= 0:
        raise ValueError("size must be positive")

    if method == "shuffle":
        warnings.warn(
            "method 'shuffle' is deprecated, please use 'boyett'", FutureWarning
        )
        method = "boyett"

    rng = _util.normalize_rng(random_state)

    w = np.array(w, dtype=float)
    if w.ndim != 2:
        raise ValueError("w must be two-dimensional")

    r = np.sum(w, axis=1)
    c = np.sum(w, axis=0)
    ntot = np.sum(r)

    m = np.outer(r, c) / ntot

    f1 = 1.0 / (ntot * (ntot - 3))
    f2 = 4.0 / (ntot * (ntot - 2) * (ntot - 3))

    t = _usp(f1, f2, w, m)

    ts = np.empty(size)
    for b, w in enumerate(
        _stats.random_table(r, c).rvs(
            size, method=None if method == "auto" else method, random_state=rng
        )
    ):
        # m stays the same, since r and c remain unchanged
        ts[b] = _usp(f1, f2, w, m)

    # Thomas B. Berrett, Ioannis Kontoyiannis, Richard J. Samworth
    # Ann. Statist. 49(5): 2457-2490 (October 2021). DOI: 10.1214/20-AOS2041
    # Eq. 5 says we need to add 1 to n_pass and n_total
    pvalue = (np.sum(t <= ts) + 1) / (size + 1)

    return TestResult(t, pvalue, ts)


def _usp(f1: float, f2: float, w: NDArray, m: NDArray) -> NDArray:
    # Eq. 2.1 from https://doi.org/10.1098/rspa.2021.0549
    return f1 * np.sum((w - m) ** 2) - f2 * np.sum(w * m)


def same_population(
    fn: Callable[..., float],
    x: "ArrayLike",
    y: "ArrayLike",
    *args: "ArrayLike",
    transform: Optional[Callable[[NDArray], NDArray]] = None,
    size: int = 9999,
    random_state: Optional[Union[np.random.Generator, int]] = None,
) -> TestResult:
    """
    Compute p-value for hypothesis that samples originate from same population.

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
    size : int, optional
        Number of permutations. Default 9999.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Returns
    -------
    TestResult

    """
    if size <= 0:
        raise ValueError("max_size must be positive")

    rng = _util.normalize_rng(random_state)

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
    ts = np.empty(size)
    for b in range(size):
        rng.shuffle(joined_sample)
        ts[b] = fn(*(joined_sample[sl] for sl in slices))
    if transform is None:
        u = t
        us = ts
    else:
        u = transform(t)
        us = transform(ts)
    # see usp for why we need to add 1 to both numerator and denominator
    pvalue = (np.sum(u <= us) + 1) / (size + 1)

    return TestResult(t, pvalue, ts)


def anova(
    x: "ArrayLike", y: "ArrayLike", *args: "ArrayLike", **kwargs: Any
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
    x: "ArrayLike", y: "ArrayLike", *args: "ArrayLike", **kwargs: Any
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


def pearsonr(x: "ArrayLike", y: "ArrayLike", **kwargs: Any) -> TestResult:
    """
    Test whether two samples are drawn from same population using correlation.

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


def spearmanr(x: "ArrayLike", y: "ArrayLike", **kwargs: Any) -> TestResult:
    """
    Test whether two samples are drawn from same population using rank correlation.

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


def ttest(x: "ArrayLike", y: "ArrayLike", **kwargs: Any) -> TestResult:
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


def _ttest(x: NDArray, y: NDArray) -> float:
    n1 = len(x)
    n2 = len(y)
    m1 = np.mean(x)
    m2 = np.mean(y)
    v1 = np.var(x, ddof=1)
    v2 = np.var(y, ddof=1)
    r: float = (m1 - m2) / np.sqrt(v1 / n1 + v2 / n2)
    return r


def _pearson(x: NDArray, y: NDArray) -> float:
    m1 = np.mean(x)
    m2 = np.mean(y)
    s1 = np.mean((x - m1) ** 2)
    s2 = np.mean((y - m2) ** 2)
    r: float = np.mean((x - m1) * (y - m2)) / np.sqrt(s1 * s2)
    return r


def _spearman(x: NDArray, y: NDArray) -> float:
    x = _stats.rankdata(x)
    y = _stats.rankdata(y)
    return _pearson(x, y)


def _kruskal(*args: NDArray) -> float:
    # see https://en.wikipedia.org/wiki/
    #           Kruskal%E2%80%93Wallis_one-way_analysis_of_variance
    # method 3 and 4
    joined = np.concatenate(args)
    r = _stats.rankdata(joined)
    n = len(r)
    start = 0
    r_args = []
    for i, a in enumerate(args):
        r_args.append(r[start : start + len(a)])
        start += len(a)

    # method 3 (assuming no ties)
    h: float = 12.0 / (n * (n + 1)) * sum(
        len(r) * np.mean(r) ** 2 for r in r_args
    ) - 3 * (n + 1)

    # apply tie correction
    h /= _stats.tiecorrect(r)
    return h


class _ANOVA:
    # see https://en.wikipedia.org/wiki/F-test
    km1: int = -2
    nmk: int = 0
    a_bar: float = 0.0

    def __call__(self, *args: NDArray) -> float:
        if self.km1 == -2:
            self._init(args)

        between_group_variability: float = (
            sum(len(a) * (np.mean(a) - self.a_bar) ** 2 for a in args) / self.km1
        )
        within_group_variability: float = sum(len(a) * np.var(a) for a in args) / (
            self.nmk
        )
        return between_group_variability / within_group_variability

    def _init(self, args: Tuple[NDArray, ...]) -> None:
        n = sum(len(a) for a in args)
        k = len(args)
        self.km1 = k - 1
        self.nmk = n - k
        self.a_bar = np.mean(np.concatenate(args))
