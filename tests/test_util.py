from resample._ext import rcond2
import numpy as np


def test_rcond2():
    m = np.empty((3, 2))
    c = np.array([1.0, 2.0])
    r = np.array([3.0, 4.0, 5.0])

    rcond2(m, r, c)
