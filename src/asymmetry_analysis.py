import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# read merged data
df = pd.read_csv(
    "data/processed/final_data.csv"
)

# divide rise and fall
rise = df[
    df["oil_change"] > 0
]["actual_adjustment"]

fall = df[
    df["oil_change"] < 0
]["actual_adjustment"]

# print mean values
print("\n===== asymmetry analysis =====\n")

print(
    "Average adjustment when oil price rises:",
    rise.mean()
)

print(
    "Average adjustment when oil price falls:",
    fall.mean()
)

# t-test
t_stat, p_value = ttest_ind(
    rise,
    fall
)

print("\nt-statistic:", t_stat)

print("p-value:", p_value)

# boxplot
plt.boxplot(
    [rise, fall],
    labels=["Rise", "Fall"]
)

plt.title(
    "Asymmetric Adjustment"
)

plt.ylabel("Adjustment")
plt.savefig(
    "figures/asymmetry_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()