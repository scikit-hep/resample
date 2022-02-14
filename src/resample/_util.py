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


def rcont(
    n: int, r: np.ndarray, c: np.ndarray, method: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate random matrices conditional on row and column sum."""
    nr = len(r)
    nc = len(c)
    w = np.empty((n, nr, nc))
    _ext.rcont(w, r, c, method, rng)
    return w
