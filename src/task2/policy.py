import matplotlib.pyplot as plt
import numpy as np
states = [-2, -1, 0, 1, 2]
policy = np.array([1, 1, 0, 1, 1])

plt.figure(figsize=(6,4))
plt.bar(states, policy)

plt.title("Optimal Policy from MDP Value Iteration")
plt.xlabel("Oil Price Change State")
plt.ylabel("Optimal Action")

plt.axhline(0, color='black', linewidth=1)

plt.tight_layout()
plt.savefig("figures/policy.png", dpi=300)
plt.show()