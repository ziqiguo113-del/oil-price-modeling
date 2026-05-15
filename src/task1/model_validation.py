import pandas as pd
import matplotlib.pyplot as plt

# read result
df = pd.read_csv(
    "merged_result.csv"
)

# convert date
df["date"] = pd.to_datetime(
    df["date"]
)

# plot
plt.figure(figsize=(14,6))

plt.plot(
    df["date"],
    df["actual_adjustment"],
    marker="o",
    label="actual"
)

plt.plot(
    df["date"],
    df["theoretical_adjustment"],
    marker="s",
    label="theoretical"
)

plt.title(
    "Actual vs Theoretical Adjustment"
)

plt.xlabel("Date")

plt.ylabel("Adjustment")

plt.legend()

plt.grid(True)

# rotate dates
plt.xticks(rotation=45)

# save
plt.savefig(
    "output/adjustment_compare.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()