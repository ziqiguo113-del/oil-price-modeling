import numpy as np
import pandas as pd

from reward_function import calculate_reward

# =========================================
# 1. action space
# =========================================

actions = [
    -400, -300, -200, -100, -50,
    0,
    50, 100, 200, 300, 400, 500
]

# =========================================
# 2. state space
# =========================================

dirs = [-2, -1, 0, 1, 2]

cum_states = [
    -500, -400, -300, -200, -100,
    -50,
    0,
    50, 100, 200, 300, 400, 500
]

econ_states = [0, 1]

states = []

for d in dirs:
    for c in cum_states:
        for e in econ_states:

            states.append(
                (d, c, e)
            )

n_states = len(states)

print(f"Total states = {n_states}")

# =========================================
# 3. load transition matrix
# =========================================

P_dir = np.loadtxt(
    "transition_matrix.csv",
    delimiter=","
)

# =========================================
# 4. theoretical adjustment map
# =========================================

theoretical_map = {
    -2: -748,
    -1: -249,
     0: 0,
     1: 298,
     2: 1004
}

# =========================================
# 5. helper function
# =========================================

def discretize_cum(x):

    """
    map cumulative untreated amount
    back to nearest discrete state
    """

    candidates = np.array([
        -500, -400, -300, -200, -100,
        -50,
        0,
        50, 100, 200, 300, 400, 500
    ])

    x = max(-500, min(500, x))

    nearest = candidates[
        np.argmin(
            np.abs(candidates - x)
        )
    ]

    return int(nearest)

# =========================================
# 6. initialize value function
# =========================================

gamma = 0.90

theta = 1e-3

max_iterations = 500

V = np.zeros(n_states)

policy = np.zeros(
    n_states,
    dtype=int
)

# =========================================
# 7. value iteration
# =========================================

iteration = 0

while True:

    delta = 0

    new_V = np.copy(V)

    # =====================================
    # loop over all states
    # =====================================

    for i, state in enumerate(states):

        d, c, e = state

        # =================================
        # theoretical adjustment
        # =================================

        delta_theory = theoretical_map[d]

        best_value = -1e18

        best_action = 0

        # =================================
        # loop over all actions
        # =================================

        for a in actions:

            # =============================
            # reward
            # =============================

            reward = calculate_reward(
                a=a,
                delta_theory=delta_theory,
                econ=e,
                P_intl=80,
                a_prev=0
            )

            # =============================
            # update cumulative untreated
            # =============================

            new_cum = discretize_cum(c + delta_theory - a)

            # =============================
            # expected future value
            # only dir follows Markov chain
            # =============================

            future_value = 0

            current_dir_index = dirs.index(d)

            for next_dir_index, prob in enumerate(
                P_dir[current_dir_index]
            ):

                next_dir = dirs[
                    next_dir_index
                ]

                next_state = (
                    next_dir,
                    new_cum,
                    e
                )

                next_state_index = states.index(
                    next_state
                )

                future_value += (
                    prob
                    * V[next_state_index]
                )

            # =============================
            # Bellman update
            # =============================

            total_value = (
                reward
                + gamma * future_value
            )

            # =============================
            # update best action
            # =============================

            if total_value > best_value:

                best_value = total_value

                best_action = a

        # =================================
        # update state value
        # =================================

        new_V[i] = best_value

        policy[i] = best_action

        delta = max(
            delta,
            abs(V[i] - new_V[i])
        )

    # =====================================
    # overwrite value function
    # =====================================

    V = new_V

    iteration += 1

    print(
        f"Iteration {iteration}, "
        f"delta = {delta:.6f}"
    )

    # =====================================
    # convergence check
    # =====================================

    if delta < theta:

        print("\nValue Iteration Converged!")

        break

    if iteration >= max_iterations:

        print("\nReached Max Iterations!")

        break

# =========================================
# 8. print optimal policy
# =========================================

print(
    "\n===== Optimal Policy =====\n"
)

policy_rows = []

for i, state in enumerate(states):

    d, c, e = state

    action = int(policy[i])

    print(
        f"dir={d}, "
        f"cum={c}, "
        f"econ={e} "
        f"-> action={action}"
    )

    policy_rows.append({

        "dir": d,
        "cum": c,
        "econ": e,
        "best_action": action
    })

# =========================================
# 9. save policy csv
# =========================================

policy_df = pd.DataFrame(
    policy_rows
)

policy_df.to_csv(
    "policy_基准.csv",
    index=False
)

print(
    "\nPolicy saved as "
    "policy_基准.csv"
)

# =========================================
# 10. save value function
# =========================================

value_df = pd.DataFrame({

    "state": [
        str(s)
        for s in states
    ],

    "value": V
})

value_df.to_csv(
    "value_function.csv",
    index=False
)

print(
    "Value function saved as "
    "value_function.csv"
)