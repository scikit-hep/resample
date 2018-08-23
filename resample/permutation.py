import numpy as np


def ttest(a1, a2, b=1000, dropna=True):
    """
    Perform permutation two sample t-test.  The
    t-statistic is calculated for the two samples
    which are then pooled.  b permutations of the
    pooled data taken and the statistic is recalculated
    for each permutation of the data.  The statistic as
    well as the proportion of the permutation distribution
    less than or equal to that statistic are returned.

    Parameters
    ----------
    a1 : array-like
        First sample
    a2 : array-like
        Second sample
    b : int
        Number of permutations

    Returns
    -------
    stat : float
        t statistic
    prop : float
        Proportion of permutation distribution less than
        or equal to t

    """
    def g(x, y):
        return ((np.mean(x) - np.mean(y)) /
                np.sqrt(np.var(x) / len(x) + np.var(y) / len(y)))

    if dropna:
        a1 = a1[~np.isnan(a1)]
        a2 = a1[~np.isnan(a2)]

    t = g(a1, a2)

    n1 = len(a1)
    n2 = len(a2)

    X = np.reshape(np.tile(np.append(a1, a2), b), newshape=(b, n1 + n2))
    X = np.apply_along_axis(func1d=np.random.permutation, arr=X, axis=1)
    permute_t = np.apply_along_axis(func1d=lambda s: g(s[:n1], s[n1:]),
                                    arr=X,
                                    axis=1)

    return {"stat": t, "prop": np.mean(permute_t <= t)}
