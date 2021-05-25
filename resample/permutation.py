"""
Permutation-based equivalence tests
===================================

A collection of statistical tests of the null hypothesis that two or more samples are
compatible. The tests check either whether the samples have compatible means or are
drawn from the same population. Permutations are used to compute the distribution of
each test statistic under the null hypothesis, which gives accurate p-values without
relying on approximate formulas which are only exact in the asymptotic limit of large
samples.

The tests return a PermutationResult object, which mimics the interface of the result
objects returned by tests in scipy.stats.

Further reading:
https://en.wikipedia.org/wiki/P-value
https://en.wikipedia.org/wiki/Test_statistic
https://en.wikipedia.org/wiki/Paired_difference_test
"""

from typing import Callable, Generator, Iterable, List, Optional, Sized, Tuple, Union

import numpy as np
from scipy.stats import rankdata, tiecorrect

from resample.empirical import cdf_gen


class PermutationResult(Sized, Iterable):
    """Holder of the result of the permutation test."""

    __slots__ = ("_statistic", "_pvalue", "_samples")

    def __init__(self, statistic: float, pvalue: float, samples: np.ndarray):
        self._statistic = statistic
        self._pvalue = pvalue
        self._samples = samples

    @property
    def statistic(self) -> float:
        """Value of the test statistic computed on the original data."""
        return self._statistic

    @property
    def pvalue(self) -> float:
        """Chance probability (aka Type I error) for rejecting the null hypothesis.

        This calculates the chance probability to get a value of the test statistic
        at least as extreme as the actual value if the null hpyothesis is true. See
        https://en.wikipedia.org/wiki/P-value for details.

        Notes
        -----
        The p-value is computed like the type I error rate, but the two are conceptually
        distinct. The p-value is a random number obtained from a sample, while the type
        I error rate is a property of the test based on the p-value. Part of the test
        description is to reject the null hypothesis if the p-value is smaller than a
        probability alpha. This alpha has to be fixed before the test is carried out.
        Then, if the p-value is computed correctly, the test has a type I error rate of
        at most alpha.
        """
        return self._pvalue

    @property
    def samples(self) -> np.ndarray:
        """Values of the test statistic from the permutated samples."""
        return self._samples

    def __repr__(self) -> str:
        s = None
        if len(self.samples) < 7:
            s = str(self.samples)
        else:
            s = "[{0} {1} {2} ... {3} {4} {5}]".format(
                *self.samples[:3], *self.samples[3:]
            )
        return "<PermutationResult statistic={0} pvalue={1} samples={2}>".format(
            self.statistic, self.pvalue, s
        )

    def __iter__(self) -> Generator:
        for i in range(3):
            yield self[i]

    def __len__(self) -> int:
        return 3

    def __getitem__(self, i: int) -> Union[float, np.ndarray]:
        if i == 0:
            return self.statistic
        elif i == 1:
            return self.pvalue
        elif i == 2:
            return self.samples
        raise IndexError


def ttest(
    a1: Iterable,
    a2: Iterable,
    size: int = 1000,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> PermutationResult:
    """
    Test whether the means of two samples are compatible with Welch's t-test.

    See https://en.wikipedia.org/wiki/Welch%27s_t-test for details on this test. The
    p-value computed is for the null hypothesis that the two population means are equal.
    The test is two-sided, which means that swapping a1 and a2 gives the same pvalue.
    Welch's t-test does not require the sample sizes to be equal and it does not require
    the samples to have the same variance.

    Parameters
    ----------
    a1 : array-like
        First sample.
    a2 : array-like
        Second sample.
    size : int, optional
        Number of permutations. Default 1000.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Returns
    -------
    PermutationResult
    """
    t, ts = _compute_statistics(_ttest, (a1, a2), size, random_state)
    return PermutationResult(t, np.mean(np.abs(ts) > np.abs(t)), ts)


def anova(
    *args: Iterable,
    size: int = 1000,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> PermutationResult:
    """
    Test whether the means of two or more samples are compatible.

    This test uses one-way analysis of variance (one-way ANOVA), see
    https://en.wikipedia.org/wiki/One-way_analysis_of_variance and
    https://en.wikipedia.org/wiki/F-test for details. This test is typically used when
    one has three groups or more. For two groups, Welch's ttest is preferred, because
    ANOVA assumes equal variances for the samples.

    Parameters
    ----------
    args : sequence of array-like
        Samples.
    size : int, optional
        Number of permutations. Default 1000.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Returns
    -------
    PermutationResult
    """
    t, ts = _compute_statistics(_ANOVA(), args, size, random_state)
    return PermutationResult(t, np.mean(ts > t), ts)


def mannwhitneyu(
    a1: Iterable,
    a2: Iterable,
    size: int = 1000,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> PermutationResult:
    """
    Test whether two samples are drawn from the same population based on ranking.

    This performs the permutation-based Mann-Whitney U test, see
    https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test for details. The test
    works for any population of samples that are ordinal, so also for integers. This is
    the two-sided version of the test, meaning that the test gives the same pvalue if
    a1 and a2 are swapped.

    For normally distributed data, the test is almost as powerful as the t-test and
    considerably more powerful for non-normal populations. It should be kept in mind,
    however, that the two tests are not equivalent. The t-test tests whether
    two samples have the same mean and ignores differences in variance, while the
    Mann-Whitney U test would detect differences in variance.

    Parameters
    ----------
    a1 : array-like
        First sample.
    a2 : array-like
        Second sample.
    size : int, optional
        Number of permutations. Default 1000.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Returns
    -------
    PermutationResult
    """
    t, ts = _compute_statistics(_mannwhitneyu, (a1, a2), size, random_state)
    return PermutationResult(t, np.mean(ts < t), ts)


def kruskal(
    *args: Iterable,
    size: int = 1000,
    random_state: Optional[Union[int, np.random.Generator]] = None
) -> PermutationResult:
    """
    Test whether two or more samples are drawn from the same population.

    This performs a permutation-based Kruskal-Wallis test, see
    https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance
    for details. It extends the Mann-Whitney U test to more than two groups.

    Parameters
    ----------
    args : sequence of array-like
        Samples.
    size : int, optional
        Number of permutations. Default 1000.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Returns
    -------
    PermutationResult
    """
    t, ts = _compute_statistics(_kruskal, args, size, random_state)
    return PermutationResult(t, np.mean(np.abs(ts) > np.abs(t)), ts)


def pearson(
    a1: Iterable,
    a2: Iterable,
    size: int = 1000,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> PermutationResult:
    """
    Perform permutation-based correlation test.

    Parameters
    ----------
    a1 : array-like
        First sample.
    a2 : array-like
        Second sample.
    size : int, optional
        Number of permutations. Default 1000.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Returns
    -------
    PermutationResult
    """
    rng = _normalize_rng(random_state)
    args = _process([a1, a2])
    if args is None:
        raise ValueError("input contains NaN")
    if len(args[0]) != len(args[1]):
        raise ValueError("a1 and a2 must have have the same length")
    if len(args[0]) < 2:
        raise ValueError("length of a1 and a2 must be at least 2.")
    t = _pearson(args)
    ts = _compute_permutations(rng, _pearson, size, args)
    return PermutationResult(t, np.mean(np.abs(ts) > np.abs(t)), ts)


def spearman(
    a1: Iterable,
    a2: Iterable,
    size: int = 1000,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> PermutationResult:
    """
    Perform permutation-based correlation test of rank data.

    Parameters
    ----------
    a1 : array-like
        First sample.
    a2 : array-like
        Second sample.
    size : int, optional
        Number of permutations. Default 1000.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Returns
    -------
    PermutationResult
    """
    rng = _normalize_rng(random_state)
    args = _process([a1, a2])
    if args is None:
        raise ValueError("input contains NaN")
    if len(args[0]) != len(args[1]):
        raise ValueError("a1 and a2 must have have the same length")
    if len(args[0]) < 2:
        raise ValueError("length of a1 and a2 must be at least 2.")
    t = _spearman(args)
    ts = _compute_permutations(rng, _spearman, size, args)
    return PermutationResult(t, np.mean(np.abs(ts) > np.abs(t)), ts)


def ks(
    a1: Iterable,
    a2: Iterable,
    size: int = 1000,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> PermutationResult:
    """
    Test whether two samples are drawn from the same population.

    This performs the permutation-based two-sided Kolmogorov-Smirnov test.

    Parameters
    ----------
    a1 : array-like
        First sample.
    a2 : array-like
        Second sample.
    size : int, optional
        Number of permutations. Default 1000.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Returns
    -------
    PermutationResult
    """
    t, ts = _compute_statistics(_KS(), (a1, a2), size, random_state)
    return PermutationResult(t, np.mean(ts) > t, ts)


def _compute_statistics(
    fn: Callable,
    args: Iterable[Iterable],
    size: int,
    random_state: Optional[Union[int, np.random.Generator]],
) -> Tuple[float, np.ndarray]:
    rng = _normalize_rng(random_state)
    args = _process(args)
    if args is None:
        raise ValueError("input contains NaN")
    t = fn(args)
    ts = _compute_permutations(rng, fn, size, args)
    return t, ts


def _normalize_rng(
    random_state: Optional[Union[int, np.random.Generator]]
) -> np.random.Generator:
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, int):
        return np.random.default_rng(random_state)
    return random_state


def _process(args: Iterable[Iterable]) -> Optional[List[np.ndarray]]:
    r = []
    for arg in args:
        a = np.array(arg)
        if np.any(np.isnan(a)):
            return None
        r.append(a)
    return r


def _compute_permutations(
    rng: np.random.Generator, fn: Callable, size: int, args: List[np.ndarray]
) -> np.ndarray:

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
        ts[i] = fn([arr[sl] for sl in slices])

    return ts


def _ttest(args: Tuple[np.ndarray, np.ndarray]) -> float:
    a1, a2 = args
    n1 = len(a1)
    n2 = len(a2)
    m1 = np.mean(a1)
    m2 = np.mean(a2)
    v1 = np.var(a1, ddof=1)
    v2 = np.var(a2, ddof=1)
    r: float = (m1 - m2) / np.sqrt(v1 / n1 + v2 / n2)
    return r


def _mannwhitneyu(args: Tuple[np.ndarray, np.ndarray]) -> float:
    a1, a2 = args
    # method 2 from Wikipedia
    n1 = len(a1)
    a = rankdata(np.concatenate(args))
    r1 = np.sum(a[:n1])
    u1: float = r1 - 0.5 * n1 * (n1 + 1)
    return u1


def _pearson(args: List[np.ndarray]) -> float:
    a1, a2 = args
    m1 = np.mean(a1)
    m2 = np.mean(a2)
    s1 = np.mean((a1 - m1) ** 2)
    s2 = np.mean((a2 - m2) ** 2)
    r: float = np.mean((a1 - m1) * (a2 - m2)) / np.sqrt(s1 * s2)
    return r


def _spearman(args: List[np.ndarray]) -> float:
    a1, a2 = args
    a1 = rankdata(a1)
    a2 = rankdata(a2)
    return _pearson([a1, a2])


# see https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance
# method 3 and 4
def _kruskal(args: List[np.ndarray]) -> float:
    joined = np.concatenate(args)

    r = rankdata(joined)
    n = len(r)

    start = 0
    for i, a in enumerate(args):
        args[i] = r[start : start + len(a)]
        start += len(a)

    # method 3 (assuming no ties)
    h = 12.0 / (n * (n + 1)) * sum(len(a) * np.mean(a) ** 2 for a in args) - 3 * (n + 1)

    # apply tie correction
    h /= tiecorrect(r)
    return h


# see https://en.wikipedia.org/wiki/F-test
class _ANOVA:
    km1: int = -2
    nmk: int = 0
    a_bar: float = 0.0

    def __call__(self, args: Tuple[np.ndarray, ...]) -> float:
        if self.km1 == -2:
            self._init(args)

        between_group_variability = (
            sum(len(a) * (np.mean(a) - self.a_bar) ** 2 for a in args) / self.km1
        )
        within_group_variability = sum(len(a) * np.var(a) for a in args) / (self.nmk)
        return between_group_variability / within_group_variability

    def _init(self, args: Tuple[np.ndarray, ...]) -> None:
        n = sum(len(a) for a in args)
        k = len(args)
        self.km1 = k - 1
        self.nmk = n - k
        self.a_bar = np.mean(np.concatenate(args))


class _KS:
    all = None

    def __call__(self, args: Tuple[np.ndarray, ...]) -> float:
        if self.all is None:
            self._init(args)
        a1, a2 = args
        f1 = cdf_gen(a1)
        f2 = cdf_gen(a2)
        r: float = np.max(f1(self.all) - f2(self.all))
        return r

    def _init(self, args: Tuple[np.ndarray, ...]) -> None:
        self.all = np.concatenate(args)
