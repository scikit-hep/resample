import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np

d = Path(__file__).parent if "__file__" in globals() else Path()

with open(d / "bench_rcont.json") as f:
    data = json.load(f)

vs = [[[], [], []], [[], [], []]]

benchs = data["benchmarks"]
for b in benchs:
    params = b["params"]
    m = params["method"]
    n = params["n"]
    k = params["k"]
    stats = b["stats"]
    val = stats["mean"]
    err = stats["stddev"] / stats["rounds"] ** 0.5
    vs[m][0].append(n)
    vs[m][1].append(k)
    vs[m][2].append(val)

fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for i, label in enumerate(("shuffle", "patefield")):
    ax[0].scatter(vs[i][0], vs[i][2], s=np.add(vs[i][1], 1), label=label)
    ax[1].scatter(vs[i][1], vs[i][2], s=10 * np.log(vs[i][0]) - 10)
ax[0].loglog()
ax[1].loglog()
ax[0].set_xlabel("N")
ax[1].set_xlabel("K")
ax[0].set_ylabel("t/sec")
plt.figlegend(loc="upper center", ncol=2, frameon=False)
