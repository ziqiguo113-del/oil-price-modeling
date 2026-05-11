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