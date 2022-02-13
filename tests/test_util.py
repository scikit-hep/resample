from resample._ext import rcont
import numpy as np


def test_rcont():
    m = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    r = np.sum(m, axis=1)
    c = np.sum(m, axis=0)

    w = np.empty_like(m)

    rng = np.random.default_rng(1)

    for _ in range(5):
        rcont(w, r, c, rng)
        print("r", r)
        print("c", c)
        print("w", w)
