# =========================================
# reward_function.py
# final stable version
# =========================================

import numpy as np

# =========================================
# global parameters
# =========================================

P_DOM = 8000

KAPPA = 0.01

MU = 0.5

WEIGHTS = [
    0.30,   # consumer
    0.25,   # firm
    0.20,   # inflation
    0.15,   # volatility
    0.10    # security
]

# =========================================
# reward function
# =========================================

def calculate_reward(
    a,
    delta_theory,
    econ,
    P_intl,
    a_prev
):

    """
    Final stable MDP reward function
    """

    # =====================================
    # 1. consumer loss
    # =====================================

    L_consumer = (
        a / P_DOM
    ) ** 2

    # =====================================
    # 2. firm loss
    # =====================================

    L_firm = (
        (a - delta_theory) / 1000
    ) ** 2

    # =====================================
    # 3. inflation loss
    # =====================================

    if econ == 1 and a > 0:

        L_inflation = (
            KAPPA
            * ((a / 500) ** 2)
        )

    else:

        L_inflation = 0

    # =====================================
    # 4. volatility loss
    # =====================================

    L_volatility = (
        ((a - a_prev) / 500) ** 2
    )

    # =====================================
    # 5. energy security loss
    # =====================================

    if P_intl > 130 and a < delta_theory:

        L_security = (
            MU
            * max(delta_theory - a, 0)
            / 1000
        )

    else:

        L_security = 0

    # =====================================
    # total loss
    # =====================================

    total_loss = (

        WEIGHTS[0] * L_consumer
        + WEIGHTS[1] * L_firm
        + WEIGHTS[2] * L_inflation
        + WEIGHTS[3] * L_volatility
        + WEIGHTS[4] * L_security
    )

    # =====================================
    # reward
    # =====================================

    reward = -total_loss

    return reward


# =========================================
# optional test block
# =========================================

if __name__ == "__main__":

    actions = [
        -400, -300, -200, -100, -50,
        0,
        50, 100, 200, 300, 400, 500
    ]

    delta_theory = 1004

    for a in actions:

        r = calculate_reward(
            a=a,
            delta_theory=delta_theory,
            econ=1,
            P_intl=120,
            a_prev=0
        )

        print(
            f"action={a:+4d}, "
            f"reward={r:.6f}"
        )