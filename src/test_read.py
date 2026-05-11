import pandas as pd

data = pd.read_csv("data/raw/brent_oil.csv")

print(data.shape)

print(data.head())

print(data.tail())