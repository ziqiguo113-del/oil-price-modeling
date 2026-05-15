import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

# =====================================
# load policy
# =====================================

df = pd.read_csv(
    "policy_基准.csv"
)

# =====================================
# features
# =====================================

X = df[
    ["dir", "cum", "econ"]
]

# =====================================
# target
# =====================================

y = df["best_action"]

# =====================================
# train decision tree
# =====================================

tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

tree.fit(X, y)

# =====================================
# prettier plot
# =====================================

plt.figure(
    figsize=(16, 8)
)

plot_tree(
    tree,

    feature_names=[
        "Oil Direction",
        "Cumulative Adj",
        "Economic State"
    ],

    class_names=[
        "-200",
        "-50",
        "0",
        "100",
        "300"
    ],

    filled=True,

    rounded=True,

    fontsize=12,

    impurity=False
)

plt.title(
    "Interpretable Oil Pricing Policy Tree",
    fontsize=20,
    pad=20
)

plt.savefig(
    "decision_tree_policy_beautified.png",
    dpi=400,
    bbox_inches="tight"
)

plt.show()