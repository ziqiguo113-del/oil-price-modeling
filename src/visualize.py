import pandas as pd
import matplotlib.pyplot as plt

# read merged data
data = pd.read_csv(
    "data/merged_result.csv"
)
data = data.sort_values("date")
# plot
plt.figure(figsize=(10, 5))

plt.plot(
    data["date"],
    data["actual_adjustment"],
    marker="o",
    label="actual"
)

plt.plot(
    data["date"],
    data["theoretical_adjustment"],
    marker="s",
    label="theoretical"
)

plt.legend()

plt.xticks(rotation=45)

plt.title("Actual vs Theoretical Adjustment")

plt.tight_layout()

plt.savefig(
    "output/adjustment_compare.png"
)

plt.show()