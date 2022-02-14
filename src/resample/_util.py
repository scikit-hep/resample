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


def extended_copy(s: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Return array with elements of s repeated k times."""
    r = np.empty(np.sum(k), dtype=s.dtype)
    m = 0
    for j, kj in enumerate(k):
        r[m : m + kj] = s[j]
        m += kj
    return r


def fill_w(w: np.ndarray, xmap: np.ndarray, ymap: np.ndarray) -> None:
    """Clear and fill w matrix."""
    w[:] = 0
    for i, j in zip(xmap, ymap):
        w[i, j] += 1


def rcont(n: int, r: np.ndarray, c: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate random matrices conditional on row and column sum."""
    nr = len(r)
    nc = len(c)
    m = np.empty((n, nr, nc))
    _ext.rcont(m, r, c, rng)
    return m


# optionally accelerate some functions with numba if there is a notable benefit
try:
    import numba as _nb

    _jit = _nb.njit(cache=True)

    extended_copy = _jit(extended_copy)
    fill_w = _jit(fill_w)  # factor 20-30 speed-up of usp test
except ImportError:
    pass
