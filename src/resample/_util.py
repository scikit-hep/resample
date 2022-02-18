import typing as _tp

import numpy as np

from ._ext import rcont  # noqa type: ignore


def normalize_rng(
    random_state: _tp.Optional[_tp.Union[int, np.random.Generator]]
) -> np.random.Generator:
    """Return normalized RNG object."""
    if random_state is None:
        return np.random.default_rng()
    elif isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def wilson_score_interval(n1, n, z):
    """Return binomial fraction and Wilson score interval."""
    p = n1 / n
    norm = 1 / (1 + z**2 / n)
    a = p + 0.5 * z**2 / n
    b = z * np.sqrt(p * (1 - p) / n + 0.25 * (z / n) ** 2)
    return p, ((a - b) * norm, (a + b) * norm)
