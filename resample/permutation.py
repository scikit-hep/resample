from __future__ import division
import numpy as np
from scipy.stats import rankdata
from resample.utils import ecdf


def ttest(a1, a2, b=100, random_state=None):
    """
    Perform permutation two sample t-test (the mean of a2
    is subtracted from that of a1)

    Parameters
    ----------
    a1 : array-like
        First sample

    a2 : array-like
        Second sample

    b : int, default : 100
        Number of permutations

    random_state : int or None, default : None
        random number seed

    Returns
    -------
    {'t', 'prop'} : {float, float}
        T statistic as well as proportion of permutation
        distribution less than or equal to that statistic
    """
    np.random.seed(random_state)

    a1 = np.asarray(a1)
    a2 = np.asarray(a2)

    a1 = a1[~np.isnan(a1)]
    a2 = a2[~np.isnan(a2)]

    def g(x, y):
        return ((np.mean(x) - np.mean(y)) /
                np.sqrt(np.var(x, ddof=1) /
                len(x) + np.var(y, ddof=1) / len(y)))

    t = g(a1, a2)

    n1 = len(a1)
    n2 = len(a2)

    X = np.apply_along_axis(func1d=np.random.permutation,
                            arr=np.reshape(np.tile(np.append(a1, a2), b),
                                           newshape=(b, n1 + n2)),
                            axis=1)

    permute_t = np.apply_along_axis(func1d=lambda s: g(s[:n1], s[n1:]),
                                    arr=X,
                                    axis=1)

    return {"t": t, "prop": np.mean(permute_t <= t)}


def anova(*args, b=100, random_state=None):
    """
    Perform permutation one way analysis of variance

    Parameters
    ----------
    args : sequence of array-like
        Samples

    b : int, default : 100
        Number of permutations

    random_state : int or None, default : None
        random number seed

    Returns
    -------
    {'f', 'prop'} : {float, float}
        F statistic as well as proportion of permutation
        distribution less than or equal to that statistic
    """
    np.random.seed(random_state)

    args = [np.asarray(a) for a in args]

    args = [a[~np.isnan(a)] for a in args]

    t = len(args)
    ns = [len(a) for a in args]
    n = np.sum(ns)
    pos = np.append(0, np.cumsum(ns))
    arr = np.concatenate(args)
    a_bar = np.mean(arr)

    def g(a):
        sse = np.sum([ns[i] * np.var(a[pos[i]:pos[i+1]]) for i in range(t)])
        ssb = (np.sum([ns[i] * (np.mean(a[pos[i]:pos[i+1]]) - a_bar)**2
                       for i in range(t)]))
        return (ssb / (t - 1)) / (sse / (n - t))

    X = np.reshape(np.tile(arr, b), newshape=(b, n))

    f = g(arr)

    permute_f = np.apply_along_axis(func1d=(lambda x:
                                            g(np.random.permutation(x))),
                                    arr=X,
                                    axis=1)

    return {"f": f, "prop": np.mean(permute_f <= f)}


def wilcoxon(a1, a2, b=100, random_state=None):
    """
    Perform permutation Wilcoxon rank sum test

    Parameters
    ----------
    a1 : array-like
        First sample

    a2 : array-like
        Second sample

    b : int, default : 100
        Number of permutations

    random_state : int or None, default : None
        random number seed

    Returns
    -------
    {'w', 'prop'} : {int, float}
        W statistic as well as proportion of permutation
        distribution less than or equal to that statistic
    """
    np.random.seed(random_state)

    a1 = np.asarray(a1)
    a2 = np.asarray(a2)

    a1 = a1[~np.isnan(a1)]
    a2 = a2[~np.isnan(a2)]

    n1 = len(a1)
    n2 = len(a2)

    a = np.append(a1, a2)
    a = rankdata(a)

    X = np.apply_along_axis(func1d=np.random.permutation,
                            arr=np.reshape(np.tile(a, b),
                                           newshape=(b, n1 + n2)),
                            axis=1)

    w = np.sum(a[:n1])

    permute_w = np.apply_along_axis(func1d=lambda s: np.sum(s[:n1]),
                                    arr=X,
                                    axis=1)

    return {"w": w, "prop": np.mean(permute_w <= w)}


def kruskal_wallis(*args, b=100, random_state=None):
    """
    Perform permutation Kruskal-Wallis test

    Parameters
    ----------
    args : sequence of array-like
        Samples

    b : int, default : 100
        Number of permutations

    random_state : int or None, default : None
        random number seed

    Returns
    -------
    {'h', 'prop'} : {float, float}
        H statistic as well as proportion of permutation
        distribution less than or equal to that statistic
    """
    np.random.seed(random_state)

    args = [np.asarray(a) for a in args]
    args = [a[~np.isnan(a)] for a in args]

    t = len(args)
    ns = [len(a) for a in args]
    n = np.sum(ns)
    pos = np.append(0, np.cumsum(ns))
    r_arr = rankdata(np.concatenate(args))
    ri_means = [np.mean(r_arr[pos[i]:pos[i+1]]) for i in range(t)]
    r_mean = np.mean(r_arr)

    def g(a):
        num = np.sum([ns[i] * (ri_means[i] - r_mean)**2 for i in range(t)])
        den = (np.sum([np.sum((r_arr[pos[i]:pos[i+1]] -
               r_mean)**2) for i in range(t)]))
        return (n - 1) * num / den

    X = np.reshape(np.tile(r_arr, b), newshape=(b, n))

    h = g(r_arr)

    permute_h = np.apply_along_axis(func1d=(lambda x:
                                            g(np.random.permutation(x))),
                                    arr=X,
                                    axis=1)

    return {"h": h, "prop": np.mean(permute_h <= h)}


def corr_test(a1, a2, method="pearson", b=100, random_state=None):
    """
    Perform permutation correlation test

    Parameters
    ----------
    a1 : array-like
        First sample

    a2 : array-like
        Second sample

    method : string, {'pearson', 'spearman'},
            default : 'pearson'
        Correlation method

    b : int, default : 100
        Number of permutations

    random_state : int or None, default : None
        random number seed

    Returns
    -------
    {'corr', 'prop'} : {float, float}
        Correlation as well as proportion of permutation
        distribution less than or equal to that statistic
    """
    np.random.seed(random_state)

    a1 = np.asarray(a1)
    a2 = np.asarray(a2)

    n1 = len(a1)
    n2 = len(a2)

    if n1 != n2:
        raise ValueError("a1 and a2 must have have"
                         " the same length")

    a = np.column_stack((a1, a2))

    a = a[np.amax(~np.isnan(a), axis=1)]

    if method in ["pearson", "distance"]:
        X = np.asarray([a] * b)
    elif method == "spearman":
        a = np.apply_along_axis(func1d=rankdata,
                                arr=a,
                                axis=0)
        X = np.asarray([a] * b)
    else:
        raise ValueError("method must be either 'pearson'"
                         " , 'spearman', or 'distance',"
                         " '{method}' was supplied".
                         format(method=method))

    def corr(x, y):
        return np.corrcoef(x, y)[0, 1]

    c = corr(a[:, 0], a[:, 1])

    permute_c = (np.asarray([corr(np.random.permutation(x[:, 0]), x[:, 1])
                             for x in X]))

    return {"c": c, "prop": np.mean(permute_c <= c)}


def ks_test(a1, a2, b=100, random_state=None):
    """
    Perform permutation two sample K-S test

    Parameters
    ----------
    a1 : array-like
        First sample

    a2 : array-like
        Second sample

    b : int, default : 100
        Number of permutations

    random_state : int or None, default : None
        random number seed

    Returns
    -------
    {'d', 'prop'} : {float, float}
        D statistic as well as proportion of permutation
        distribution less than or equal to that statistic
    """
    np.random.seed(random_state)

    a1 = np.asarray(a1)
    a2 = np.asarray(a2)

    a1 = a1[~np.isnan(a1)]
    a2 = a2[~np.isnan(a2)]

    n1 = len(a1)
    n2 = len(a2)
    n = n1 + n2

    f1 = ecdf(a1)
    f2 = ecdf(a2)
    a = np.sort(np.append(a1, a2))
    d = np.max([abs(f1(v) - f2(v)) for v in a])

    def h(arr, i, m):
        return (np.searchsorted(arr, i, side="right",
                                sorter=None) / m)

    def g(s):
        mask = np.ones(n, dtype=np.bool)
        mask[np.random.choice(range(n), size=n2, replace=False)] = False

        return (np.max([abs(h(s[mask], i, n1) - h(s[~mask], i, n2))
                        for i in s]))

    X = np.reshape(np.tile(a, b), newshape=(b, n))

    permute_d = np.apply_along_axis(func1d=g,
                                    arr=X,
                                    axis=1)

    return {"d": d, "prop": np.mean(permute_d <= d)}
