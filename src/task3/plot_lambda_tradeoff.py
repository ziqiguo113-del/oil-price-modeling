import pandas as pd
import matplotlib.pyplot as plt

# load result
df = pd.read_csv("lambda_results.csv")

# plot
plt.figure(figsize=(6, 4))

plt.plot(
    df["complexity"],
    df["welfare"],
    marker="o"
)

# annotate lambda
for i in range(len(df)):

    plt.text(
        df["complexity"][i],
        df["welfare"][i],
        f"λ={df['lambda'][i]}"
    )

plt.xlabel("Policy Complexity")

plt.ylabel("Welfare")

plt.title("Complexity-Welfare Tradeoff")

plt.grid(True)

plt.savefig(
    "complexity_welfare_tradeoff.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()