"""
Permutation-based equivalence tests
===================================

A collection of statistical tests of the null hypothesis that two or more samples are
compatible. More precisely, the tests check either whether the samples have compatible
means or are drawn from the same population. Permutations are used to compute the
distribution of each test statistic under the null hypothesis, which gives accurate
Type I error rates without relying on approximate formulas which are only exact in the
asymptotic limit of large samples.
"""

from typing import Dict, List

import numpy as np
from scipy.stats import rankdata

from resample.empirical import cdf_gen


def ttest(a1: np.ndarray, a2: np.ndarray, b: int = 100, random_state=None) -> Dict:
    """
    Test whether the means of two samples are compatible with Welch's t-test.

    See https://en.wikipedia.org/wiki/Welch%27s_t-test for details on this test. This
    implementation does not require the sample sizes to be equal and it does not require
    the samples to have the same variance. NaN values are filtered out by this
    implementation.

    Parameters
    ----------
    a1 : array-like
        First sample.
    a2 : array-like
        Second sample.
    b : int, optional
        Number of permutations. Default 100.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Returns
    -------
    {'t': float, 'prop': float}
        T statistic as well as proportion of permutation distribution less than or
        equal to that statistic.
    """
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    a1 = np.asarray(a1)
    a2 = np.asarray(a2)

    a1 = a1[~np.isnan(a1)]
    a2 = a2[~np.isnan(a2)]

    def g(x, y):
        return (np.mean(x) - np.mean(y)) / np.sqrt(
            np.var(x, ddof=1) / len(x) + np.var(y, ddof=1) / len(y)
        )

    t = g(a1, a2)

    n1 = len(a1)
    n2 = len(a2)

    X = np.apply_along_axis(
        func1d=rng.permutation,
        arr=np.reshape(np.tile(np.append(a1, a2), b), newshape=(b, n1 + n2)),
        axis=1,
    )

    permute_t = np.apply_along_axis(func1d=lambda s: g(s[:n1], s[n1:]), arr=X, axis=1)

    return {"t": t, "prop": np.mean(permute_t <= t)}


def anova(args: List[np.ndarray], b: int = 100, random_state=None) -> Dict:
    """
    Test whether the means of two or more samples are compatible.

    This test uses one-way analysis of variance (one-way ANOVA), see
    https://en.wikipedia.org/wiki/One-way_analysis_of_variance for details. This test is
    typically used when one has three groups or more. For two groups, the test is
    equivalent to the t-test. NaN values are filtered out by this implementation.

    Parameters
    ----------
    args : sequence of array-like
        Samples.
    b : int, optional
        Number of permutations. Default 100.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Returns
    -------
    {'f': float, 'prop': float}
        F statistic as well as proportion of permutation distribution less than or
        equal to that statistic.
    """
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    args = [np.asarray(a) for a in args]
    args = [a[~np.isnan(a)] for a in args]

    t = len(args)
    ns = [len(a) for a in args]
    n = np.sum(ns)
    pos = np.append(0, np.cumsum(ns))
    arr = np.concatenate(args)
    a_bar = np.mean(arr)

    def g(a):
        sse = np.sum(
            [ns[i] * np.var(a[pos[i] : pos[i + 1]]) for i in range(t)]  # noqa: E203
        )
        ssb = np.sum(
            [
                ns[i] * (np.mean(a[pos[i] : pos[i + 1]]) - a_bar) ** 2  # noqa: E203
                for i in range(t)
            ]
        )
        return (ssb / (t - 1)) / (sse / (n - t))

    X = np.reshape(np.tile(arr, b), newshape=(b, n))

    f = g(arr)

    permute_f = np.apply_along_axis(
        func1d=(lambda x: g(rng.permutation(x))), arr=X, axis=1
    )

    return {"f": f, "prop": np.mean(permute_f <= f)}


def wilcoxon(a1: np.ndarray, a2: np.ndarray, b: int = 100, random_state=None) -> Dict:
    """
    Test whether two samples are drawn from the same population.

    This performs the permutation-based Wilcoxon rank sum test, also known as
    Mann-Whitney U test, see https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test
    for details.

    Parameters
    ----------
    a1 : array-like
        First sample.
    a2 : array-like
        Second sample.
    b : int, optional
        Number of permutations. Default 100.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Returns
    -------
    {'w': float, 'prop': float}
        W statistic as well as proportion of permutation distribution less than or
        equal to that statistic.
    """
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    a1 = np.asarray(a1)
    a2 = np.asarray(a2)

    a1 = a1[~np.isnan(a1)]
    a2 = a2[~np.isnan(a2)]

    n1 = len(a1)
    n2 = len(a2)

    a = np.append(a1, a2)
    a = rankdata(a)

    X = np.apply_along_axis(
        func1d=rng.permutation,
        arr=np.reshape(np.tile(a, b), newshape=(b, n1 + n2)),
        axis=1,
    )

    w = np.sum(a[:n1])

    permute_w = np.apply_along_axis(func1d=lambda s: np.sum(s[:n1]), arr=X, axis=1)

    return {"w": w, "prop": np.mean(permute_w <= w)}


def kruskal_wallis(args: List[np.ndarray], b: int = 100, random_state=None) -> Dict:
    """
    Test whether two or more samples are drawn from the same population.

    This performs the permutation-based Kruskal-Wallis test, see
    https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance
    for details.

    Parameters
    ----------
    args : sequence of array-like
        Samples.
    b : int, optional
        Number of permutations. Default 100.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Returns
    -------
    {'h': float, 'prop': flaot}
        H statistic as well as proportion of permutation distribution less than or
        equal to that statistic.
    """
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    args = [np.asarray(a) for a in args]
    args = [a[~np.isnan(a)] for a in args]

    t = len(args)
    ns = [len(a) for a in args]
    n = np.sum(ns)
    pos = np.append(0, np.cumsum(ns))
    r_arr = rankdata(np.concatenate(args))
    ri_means = [np.mean(r_arr[pos[i] : pos[i + 1]]) for i in range(t)]  # noqa: E203
    r_mean = np.mean(r_arr)

    def g(a):
        num = np.sum([ns[i] * (ri_means[i] - r_mean) ** 2 for i in range(t)])
        den = np.sum(
            [
                np.sum((a[pos[i] : pos[i + 1]] - r_mean) ** 2)  # noqa: E203
                for i in range(t)
            ]
        )
        return (n - 1) * num / den

    x = np.reshape(np.tile(r_arr, b), newshape=(b, n))

    h = g(r_arr)

    permute_h = np.apply_along_axis(
        func1d=(lambda s: g(rng.permutation(s))), arr=x, axis=1
    )

    return {"h": h, "prop": np.mean(permute_h > h)}


def corr_test(
    a1: np.ndarray,
    a2: np.ndarray,
    method: str = "pearson",
    b: int = 100,
    random_state=None,
) -> Dict:
    """
    Perform permutation correlation test.

    Parameters
    ----------
    a1 : array-like
        First sample.
    a2 : array-like
        Second sample.
    method : str, {'pearson', 'spearman'}, optional
        Correlation method. Default 'pearson'.
    b : int, optional
        Number of permutations. Default 100.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Returns
    -------
    {'corr': float, 'prop': float}
        Correlation as well as proportion of permutation distribution less than or
        equal to that statistic.
    """
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    a1 = np.asarray(a1)
    a2 = np.asarray(a2)

    n1 = len(a1)
    n2 = len(a2)

    if n1 != n2:
        raise ValueError("a1 and a2 must have have the same length")

    a = np.column_stack((a1, a2))

    a = a[np.amax(~np.isnan(a), axis=1)]

    if method in ["pearson", "distance"]:
        X = np.asarray([a] * b)
    elif method == "spearman":
        a = np.apply_along_axis(func1d=rankdata, arr=a, axis=0)
        X = np.asarray([a] * b)
    else:
        raise ValueError(
            "method must be either 'pearson', "
            "'spearman', or 'distance', "
            f"'{method}' was supplied"
        )

    def corr(x, y):
        return np.corrcoef(x, y)[0, 1]

    c = corr(a[:, 0], a[:, 1])

    permute_c = np.asarray([corr(rng.permutation(x[:, 0]), x[:, 1]) for x in X])

    return {"c": c, "prop": np.mean(permute_c <= c)}


def ks_test(a1: np.ndarray, a2: np.ndarray, b: int = 100, random_state=None) -> Dict:
    """
    Test whether two samples are drawn from the same population.

    This performs the permutation-based Kolmogorov-Smirnov test.

    Parameters
    ----------
    a1 : array-like
        First sample.
    a2 : array-like
        Second sample.
    b : int, optional
        Number of permutations. Default 100.
    random_state : numpy.random.Generator or int, optional
        Random number generator instance. If an integer is passed, seed the numpy
        default generator with it. Default is to use `numpy.random.default_rng()`.

    Returns
    -------
    {'d': float, 'prop': float}
        D statistic as well as proportion of permutation distribution less than or
        equal to that statistic.
    """
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    a1 = np.asarray(a1)
    a2 = np.asarray(a2)

    a1 = a1[~np.isnan(a1)]
    a2 = a2[~np.isnan(a2)]

    n1 = len(a1)
    n2 = len(a2)
    n = n1 + n2

    f1 = cdf_gen(a1)
    f2 = cdf_gen(a2)
    a = np.sort(np.append(a1, a2))
    d = np.max([abs(f1(v) - f2(v)) for v in a])

    def h(arr, i, m):
        return np.searchsorted(arr, i, side="right", sorter=None) / m

    def g(s):
        mask = np.ones(n, dtype=np.bool)
        mask[rng.choice(range(n), size=n2, replace=False)] = False

        return np.max([abs(h(s[mask], i, n1) - h(s[~mask], i, n2)) for i in s])

    x = np.reshape(np.tile(a, b), newshape=(b, n))

    permute_d = np.apply_along_axis(func1d=g, arr=x, axis=1)

    return {"d": d, "prop": np.mean(permute_d > d)}
