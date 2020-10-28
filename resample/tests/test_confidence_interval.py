from resample.bootstrap import confidence_interval as ci
import numpy as np

def estimator(data):
    return int(np.mean(data))

data = (1, 2, 3)
print(ci(estimator, data))