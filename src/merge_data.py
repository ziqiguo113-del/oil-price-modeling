import pandas as pd

# read oil price data
oil = pd.read_csv(
    "data/processed/processed_brent.csv"
)

# read domestic adjustment data
adjustment = pd.read_csv(
    "data/domestic_adjustment.csv"
)

# convert date
oil["date"] = pd.to_datetime(
    oil["date"]
)

adjustment["date"] = pd.to_datetime(
    adjustment["date"]
)

# merge by date
df = pd.merge(
    oil,
    adjustment,
    on="date",
    how="inner"
)

# save merged data
df.to_csv(
    "data/processed/final_data.csv",
    index=False
)

print("\nmerge complete")

print(df.head())