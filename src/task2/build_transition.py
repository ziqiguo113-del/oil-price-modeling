import pandas as pd
import numpy as np

# read data
df = pd.read_csv(
    "merged_result.csv"
)

# ======================
# classify direction
# ======================

def classify(x):

    if x < -4:
        return -2

    elif x < -1:
        return -1

    elif x <= 1:
        return 0

    elif x <= 4:
        return 1

    else:
        return 2
# create dir column
df["dir"] = df[
    "oil_change"
].apply(classify)

# state mapping
state_map = {
    -2: 0,
    -1: 1,
     0: 2,
     1: 3,
     2: 4
}

# ======================
# build count matrix
# ======================

count = np.zeros((5, 5))

dirs = df["dir"].values

for i in range(len(dirs) - 1):

    current_state = state_map[
        dirs[i]
    ]

    next_state = state_map[
        dirs[i + 1]
    ]

    count[
        current_state,
        next_state
    ] += 1

# ======================
# normalize
# ======================

P = np.zeros((5, 5))

for i in range(5):

    row_sum = count[i].sum()

    if row_sum == 0:

        P[i] = np.ones(5) / 5

    else:

        P[i] = count[i] / row_sum

# print result
print("\n===== Transition Matrix =====\n")

print(P)

# save
np.savetxt(
    "transition_matrix.csv",
    P,
    delimiter=","
)