import numpy as np

from reward_function import (
    reward_function,
    actions
)

# ======================
# 1. load transition matrix
# ======================

P = np.loadtxt(
    "transition_matrix.csv",
    delimiter=","
)

# ======================
# 2. define states
# ======================

states = [
    -2,
    -1,
    0,
    1,
    2
]

# theoretical adjustment
# different oil states correspond
# to different theoretical price changes

theoretical_map = {

    -2: -800,
    -1: -300,
     0: 0,
     1: 500,
     2: 2205
}

# ======================
# 3. initialize value function
# ======================

V = np.zeros(
    len(states)
)

gamma = 0.9

theta = 1e-3

policy = np.zeros(
    len(states)
)

# ======================
# 4. value iteration
# ======================

while True:

    delta = 0

    new_V = np.copy(V)

    for i, s in enumerate(states):

        theoretical = (
            theoretical_map[s]
        )

        action_values = []

        for a in actions:

            r = reward_function(
                theoretical,
                a
            )

            future = 0

            for j in range(
                len(states)
            ):

                future += (
                    P[i][j]
                    * V[j]
                )

            total = (
                r
                + gamma * future
            )

            action_values.append(
                total
            )

        best_value = max(
            action_values
        )

        best_action = actions[
            np.argmax(
                action_values
            )
        ]

        new_V[i] = best_value

        policy[i] = best_action

        delta = max(
            delta,
            abs(V[i] - new_V[i])
        )

    V = new_V

    if delta < theta:
        break

# ======================
# 5. output result
# ======================

print(
    "\n===== Optimal Policy =====\n"
)

for i, s in enumerate(states):

    print(
        f"state {s}: "
        f"best action ratio = "
        f"{policy[i]:.2f}"
    )