import typing as _tp

import numpy as np


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


# def patefield(n: int, r: np.ndarray, c: np.ndarray, rng) -> np.ndarray:
#     ntotal = np.sum(r)
#     assert ntotal == np.sum(c)
#
#     nr = len(r)
#     nc = len(c)
#     matrix = np.empty((nr, nc))
#     jwork = np.empty(nc - 1)
#
#     # TODO: handle cases where r or c have zero entries


# optionally accelerate some functions with numba if there is a notable benefit
try:
    import numba as _nb

    _jit = _nb.njit(cache=True)

    extended_copy = _jit(extended_copy)
    fill_w = _jit(fill_w)  # factor 20-30 speed-up of usp test
except ImportError:
    pass
