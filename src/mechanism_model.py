import pandas as pd

# read oil price data
oil = pd.read_csv("data/raw/brent_oil.csv")

# rename columns
oil.columns = [
    "date",
    "price",
    "open",
    "high",
    "low",
    "volume",
    "change_percent"
]

# convert date
oil["date"] = pd.to_datetime(oil["date"])

# sort by date
oil = oil.sort_values("date")

# convert price to float
oil["price"] = (
    oil["price"]
    .astype(str)
    .str.replace(",", "")
    .astype(float)
)

# calculate 10-day moving average
oil["ma10"] = (
    oil["price"]
    .rolling(10)
    .mean()
)

# calculate 10-day change
oil["oil_change"] = (
    oil["ma10"]
    .pct_change()
)

print(oil[[
    "date",
    "price",
    "ma10",
    "oil_change"
]].tail(20))
# simulate theoretical adjustment
# assume transmission coefficient

k = 10000

oil["theoretical_adjustment"] = (
    oil["oil_change"] * k
)

print("\n====================\n")

print(oil[[
    "date",
    "oil_change",
    "theoretical_adjustment"
]].tail(10))
# read domestic adjustment data
domestic = pd.read_csv(
    "data/domestic_adjustment.csv"
)

# convert date
domestic["date"] = pd.to_datetime(
    domestic["date"]
)

# merge
merged = pd.merge(
    domestic,
    oil[[
        "date",
        "oil_change",
        "theoretical_adjustment"
    ]],
    on="date",
    how="left"
)

print("\n====================\n")

print(merged)
merged.to_csv(
    "data/merged_result.csv",
    index=False
)
# read processed oil data
oil = pd.read_csv(
    "data/processed/processed_brent.csv"
)

# convert date
oil["date"] = pd.to_datetime(
    oil["date"]
)

# calculate theoretical adjustment
k = 10000

oil["theoretical_adjustment"] = (
    oil["oil_change"] * k
)

# read domestic adjustment
domestic = pd.read_csv(
    "data/domestic_adjustment.csv"
)

# convert date
domestic["date"] = pd.to_datetime(
    domestic["date"]
)

# merge
merged = pd.merge(
    domestic,
    oil[[
        "date",
        "oil_change",
        "theoretical_adjustment",
        "price"
    ]],
    on="date",
    how="left"
)

# save merged result
merged.to_csv(
    "data/merged_result.csv",
    index=False
)

print("\nmerge complete\n")

print(merged.head())