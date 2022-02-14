import typing as _tp

import numpy as np

from . import _ext  # type: ignore


def normalize_rng(
    random_state: _tp.Optional[_tp.Union[int, np.random.Generator]]
) -> np.random.Generator:
    """Normalize RNG."""
    if random_state is None:
        return np.random.default_rng()
    elif isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def expand(s: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Return array with elements of s repeated k times."""
    r = np.empty(np.sum(k), dtype=s.dtype)
    m = 0
    for j, kj in enumerate(k):
        r[m : m + kj] = s[j]
        m += kj
    return r


def fill(w: np.ndarray, i: int, xmap: np.ndarray, ymap: np.ndarray) -> None:
    """Clear and fill w matrix."""
    for j, k in zip(xmap, ymap):
        w[i, j, k] += 1


def rcont(
    n: int, r: np.ndarray, c: np.ndarray, method: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate random matrices conditional on row and column sum."""
    nr = len(r)
    nc = len(c)
    w = np.empty((n, nr, nc))
    if method == 0:  # auto
        method = 1  # TODO make choice based on ntot
    if method == 1:
        # Patefield algorithm
        _ext.rcont(w, r, c, rng)
    elif method == 2:
        # Naive algorithm, only kept because it is simple to implement
        xmap = expand(np.arange(nr), r.astype(np.int32))
        ymap = expand(np.arange(nc), c.astype(np.int32))
        w[:] = 0
        for i in range(n):
            rng.shuffle(ymap)
            fill(w, i, xmap, ymap)
    else:
        assert False, "invalid method"
    return w


# optionally accelerate some functions with numba if there is a notable benefit
try:
    import numba as _nb

    _jit = _nb.njit(cache=True)

    expand = _jit(expand)
    fill = _jit(fill)  # factor 20-30 speed-up of usp test
except ImportError:
    pass
