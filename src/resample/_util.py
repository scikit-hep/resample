import typing as _tp

import numpy as np


def _normalize_rng(
    random_state: _tp.Optional[_tp.Union[int, np.random.Generator]]
) -> np.random.Generator:
    if random_state is None:
        return np.random.default_rng()
    elif isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _extended_copy(s: np.ndarray, k: np.ndarray) -> np.ndarray:
    r = np.empty(np.sum(k), dtype=s.dtype)
    m = 0
    for j, kj in enumerate(k):
        r[m : m + kj] = s[j]
        m += kj
    return r


def _fill_w(w: np.ndarray, xmap: np.ndarray, ymap: np.ndarray) -> None:
    w[:] = 0
    for i, j in zip(xmap, ymap):
        w[i, j] += 1


# optionally accelerate some functions with numba if there is a notable benefit
try:
    import numba as _nb

    _jit = _nb.njit(cache=True)

    _extended_copy = _jit(_extended_copy)
    _fill_w = _jit(_fill_w)  # factor 20-30 speed-up of usp test
except ImportError:
    pass
