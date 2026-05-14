import matplotlib.pyplot as plt

from reward_function import (
    reward_function,
    actions
)

# theoretical adjustment
theoretical = 2205

# calculate reward
rewards = []

for a in actions:

    r = reward_function(
        theoretical,
        a
    )

    rewards.append(r)

# plot
plt.figure(figsize=(8,5))

plt.plot(
    actions,
    rewards,
    marker="o"
)

plt.xlabel(
    "Adjustment Ratio"
)

plt.ylabel(
    "Reward"
)

plt.title(
    "Reward Function"
)

plt.grid(True)

# save
plt.savefig(
    "figures/reward_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()