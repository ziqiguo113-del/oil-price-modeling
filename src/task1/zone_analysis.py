import pandas as pd
import matplotlib.pyplot as plt

# read data
df = pd.read_csv(
    "data/processed/final_data.csv"
)

q1 = df["price"].quantile(0.33)
q2 = df["price"].quantile(0.66)

def classify(price):

    if price < q1:
        return "low"

    elif price < q2:
        return "middle"

    else:
        return "high"

df["zone"] = df["price"].apply(classify)

# add zone column
df["zone"] = df["price"].apply(classify)

# calculate average adjustment
group = (
    df.groupby("zone")["actual_adjustment"]
    .mean()
)

print("\n===== zone analysis =====\n")

print(group)

# plot
group.plot(kind="bar")

plt.title(
    "Adjustment in Different Oil Price Zones"
)

plt.ylabel(
    "Average Adjustment"
)
plt.savefig(
    "figures/zone_analysis.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()