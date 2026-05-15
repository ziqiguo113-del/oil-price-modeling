import pandas as pd

from sklearn.tree import (
    DecisionTreeClassifier,
    export_text
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
# print rules
# =====================================

rules = export_text(
    tree,
    feature_names=[
        "dir",
        "cum",
        "econ"
    ]
)

print(
    "\n===== Decision Tree Rules =====\n"
)

print(rules)