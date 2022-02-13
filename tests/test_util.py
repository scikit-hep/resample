from resample._ext import rcond2
import numpy as np


def test_rcond2():
    m = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    r = np.sum(m, axis=1)
    c = np.sum(m, axis=0)

    rcond2(m, r, c)
    print(m)
