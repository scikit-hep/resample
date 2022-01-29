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
