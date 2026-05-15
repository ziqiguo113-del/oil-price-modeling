import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

# read merged data
data = pd.read_csv(
    "data/merged_result.csv"
)

# remove missing values
data = data.dropna()

# define X and y
X = data[["oil_change"]]

y = data["actual_adjustment"]

# build model
model = LinearRegression()

# fit model
model.fit(X, y)

# prediction
y_pred = model.predict(X)

# coefficient
k = model.coef_[0]

# intercept
b = model.intercept_

# r2
r2 = r2_score(y, y_pred)

print("\n===== regression result =====\n")

print(f"k (sensitivity): {k}")

print(f"b (intercept): {b}")

print(f"R^2 score: {r2}")