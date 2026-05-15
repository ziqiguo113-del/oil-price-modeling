import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("policy_基准.csv")

pivot = df[df["econ"] == 0].pivot(
    index="dir",
    columns="cum",
    values="best_action"
)

plt.figure(figsize=(10, 4))

sns.heatmap(
    pivot,
    annot=True,
    cmap="coolwarm"
)

plt.title("Optimal Policy Heatmap")

plt.savefig(
    "policy_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()