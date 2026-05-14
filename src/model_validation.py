import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# read data
df = pd.read_csv(
    "merged_result.csv"
)

# input and output
X = df[["theoretical_adjustment"]]

y = df["actual_adjustment"]

# build model
model = LinearRegression()

model.fit(X, y)

# prediction
pred = model.predict(X)

# evaluation
mae = mean_absolute_error(y, pred)

rmse = np.sqrt(
    mean_squared_error(y, pred)
)

print("\n===== model validation =====\n")

print("MAE:", mae)

print("RMSE:", rmse)

# plot
plt.plot(
    y.values,
    label="Actual"
)

plt.plot(
    pred,
    label="Predicted"
)

plt.legend()

plt.title(
    "Actual vs Predicted Adjustment"
)
plt.savefig(
    "figures/prediction_vs_actual.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()