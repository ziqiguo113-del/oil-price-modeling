import pandas as pd

# read raw oil price data
data = pd.read_csv(
    "data/raw/brent_oil.csv"
)

# show original columns
print(data.columns)

# rename columns
data.columns = [
    "date",
    "price",
    "open",
    "high",
    "low",
    "volume",
    "change_percent"
]

# convert date format
data["date"] = pd.to_datetime(
    data["date"]
)

# remove comma from price
data["price"] = (
    data["price"]
    .astype(str)
    .str.replace(",", "")
    .astype(float)
)

# sort by date
data = data.sort_values("date")

# reset index
data = data.reset_index(drop=True)

# calculate 10-day moving average
data["ma10"] = (
    data["price"]
    .rolling(10)
    .mean()
)

# calculate oil price change
data["oil_change"] = (
    data["ma10"]
    .pct_change()
)

# remove NaN rows
data = data.dropna()

# save processed data
data.to_csv(
    "data/processed/processed_brent.csv",
    index=False
)

print("\npreprocess complete")

print("\nprocessed data preview:\n")

print(data.head())