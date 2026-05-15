import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "task2"
        )
    )
)
import numpy as np
import pandas as pd

from reward_function import calculate_reward
# =========================================
# state space
# =========================================

dir_states = [-2, -1, 0, 1, 2]

cum_states = [
    -500, -400, -300, -200, -100,
    -50,
    0,
    50, 100, 200, 300, 400, 500
]

econ_states = [0, 1]

states = []

for d in dir_states:
    for c in cum_states:
        for e in econ_states:
            states.append((d, c, e))

# =========================================
# actions
# =========================================

actions = [
    -400, -300, -200, -100, -50,
    0,
    50, 100, 200, 300, 400, 500
]

# =========================================
# theoretical map
# =========================================

theoretical_map = {
    -2: -748,
    -1: -249,
     0: 0,
     1: 298,
     2: 1004
}

# =========================================
# helper
# =========================================

def discretize_cum(x):

    candidates = np.array([
        -500, -400, -300, -200, -100,
        -50,
        0,
        50, 100, 200, 300, 400, 500
    ])

    x = max(-500, min(500, x))

    nearest = candidates[
        np.argmin(np.abs(candidates - x))
    ]

    return int(nearest)

# =========================================
# simplified transition matrix
# =========================================

n_states = len(states)

P = np.ones(
    (n_states, n_states)
)

P = P / n_states

# =========================================
# lambda experiments
# =========================================

results = []

lambda_list = [
    0,
    0.01,
    0.03,
    0.05,
    0.10
]

for lambda_sparse in lambda_list:

    print(f"\nRunning lambda={lambda_sparse}")

    V = np.zeros(len(states))

    policy = np.zeros(len(states))

    gamma = 0.9

    theta = 1e-3

    while True:

        delta = 0

        new_V = np.copy(V)

        for i, state in enumerate(states):

            d, c, e = state

            delta_theory = theoretical_map[d]

            action_values = []

            for a in actions:

                reward = calculate_reward(
                    a=a,
                    delta_theory=delta_theory,
                    econ=e,
                    P_intl=80,
                    a_prev=0,
                    lambda_sparse=lambda_sparse
                )

                future_value = 0

                for j in range(len(states)):

                    future_value += (
                        P[i][j]
                        * V[j]
                    )

                total_value = (
                    reward
                    + gamma * future_value
                )

                action_values.append(total_value)

            best_value = max(action_values)

            best_action = actions[
                np.argmax(action_values)
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

    # =====================================
    # compute metrics
    # =====================================

    complexity = np.mean(
        np.abs(policy)
    )

    welfare = np.mean(V)

    results.append({
        "lambda": lambda_sparse,
        "complexity": complexity,
        "welfare": welfare
    })

    print(
        f"lambda={lambda_sparse}, "
        f"complexity={complexity:.2f}, "
        f"welfare={welfare:.4f}"
    )

# =========================================
# save results
# =========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "lambda_results.csv",
    index=False
)

print("\nSaved as lambda_results.csv")