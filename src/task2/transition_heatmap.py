import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read matrix
P = np.loadtxt(
    "transition_matrix.csv",
    delimiter=","
)

labels = [
    "big fall",
    "small fall",
    "stable",
    "small rise",
    "big rise"
]

plt.figure(figsize=(8,6))

plt.imshow(P)

plt.colorbar(label="probability")

plt.xticks(
    range(5),
    labels,
    rotation=30
)

plt.yticks(
    range(5),
    labels
)

plt.xlabel("next state")

plt.ylabel("current state")

plt.title(
    "Oil Price Transition Probability Matrix"
)

# show values
for i in range(5):
    for j in range(5):

        plt.text(
            j,
            i,
            f"{P[i,j]:.2f}",
            ha="center",
            va="center"
        )

plt.savefig(
    "figures/transition_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()