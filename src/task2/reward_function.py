import numpy as np

# parameters
kappa = 0.03
mu = 2

# action ratios
actions = np.linspace(
    0,
    1.1,
    12
)

def reward_function(
    theoretical,
    action_ratio
):

    # actual adjustment
    actual = (
        theoretical
        * action_ratio
    )

    # consumer loss
    consumer_loss = max(
        actual,
        0
    )

    # firm loss
    firm_loss = (
        theoretical - actual
    ) ** 2
    # CPI loss
    cpi_loss = (
        kappa
        * actual**2
    )

    # volatility loss
    volatility_loss = abs(
        actual
    )

    # security loss
    security_loss = (
        mu
        * abs(
            theoretical - actual
        )
    )

    # total reward
    reward = -(
        consumer_loss
        + firm_loss
        + cpi_loss
        + volatility_loss
        + security_loss
    )

    return reward

# test
theoretical = 2205

for a in actions:

    r = reward_function(
        theoretical,
        a
    )

    print(
        f"action={a:.1f}, reward={r:.2f}"
    )