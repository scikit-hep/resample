"""
Generate prototypes of the resample logo.

The final logo was further edited in Inkscape. The chosen version uses the Gentium Plus
Regular font from https://fontlibrary.org/en/font/gentium-plus#Gentium%20Plus-Regular.
"""

import numpy as np
from matplotlib import pyplot as plt

for font, x0 in (("ubuntu", 0.2), ("gentium", 0.17)):

    plt.figure(figsize=(5, 1.4))
    ax = plt.subplot()
    for k in ax.spines:
        ax.spines[k].set_visible(False)
    plt.tick_params(
        **{k: False for k in ax.spines}, **{f"label{k}": False for k in ax.spines}
    )
    plt.gca().set_facecolor("none")

    size = 70
    w = 0.05
    h = 0.15
    y0 = 0.1

    # original
    plt.figtext(0, y0, "re", color="r", name=font, size=size, weight="bold")
    plt.figtext(x0, y0, "sample", color="0.2", name=font, size=size)

    # copies
    rng = np.random.default_rng(1)
    s = np.fromiter("resample", "U1")
    n = 2
    for i, col in enumerate(("0.8", "0.9")):
        x = (i + 1) * w
        y = y0 + (i + 1) * h
        s2 = rng.choice(s, size=len(s))
        plt.figtext(x, y, "".join(s2), color=col, name=font, size=size, zorder=-(i + 1))

    plt.savefig(f"{font}.svg")
