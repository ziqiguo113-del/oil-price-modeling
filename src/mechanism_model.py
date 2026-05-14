import pandas as pd
import numpy as np

# ======================
# 1. read oil data
# ======================

df = pd.read_csv(
    "data/processed/processed_brent.csv"
)

# convert date
df["date"] = pd.to_datetime(df["date"])

# ======================
# 2. calculate 10-day average
# ======================

df["avg10"] = (
    df["price"]
    .rolling(10)
    .mean()
)

# ======================
# 3. read adjustment dates
# ======================

adj = pd.read_csv(
    "data/domestic_adjustment.csv"
)

adj["date"] = pd.to_datetime(
    adj["date"]
)
adj = adj.sort_values(
    "date"
)

adj = adj.reset_index(
    drop=True
)

# ======================
# 4. match adjustment windows
# ======================

result = []

# carry mechanism
carry = 0

# calibration coefficient
# 后面还能再调
k = 120

for i in range(1, len(adj)):

    current_date = adj.loc[i, "date"]

    previous_date = adj.loc[i - 1, "date"]

    current_window = df[
        (df["date"] > previous_date)
        &
        (df["date"] <= current_date)
    ]

    current_avg = (
        current_window["price"]
        .mean()
    )

# 上一期窗口
    if i >= 2:

        earlier_date = adj.loc[
            i - 2,
            "date"
        ]

        previous_window = df[
            (df["date"] > earlier_date)
            &
            (df["date"] <= previous_date)
        ]

        previous_avg = (
            previous_window["price"]
            .mean()
        )

    else:

        previous_avg = current_avg

    # oil price absolute change
    delta_price = (
        current_avg - previous_avg
    )
    # theoretical adjustment
    raw_adjustment = (
        delta_price * k
    )

    # add carry
    raw_adjustment += carry

    # 50 yuan threshold
    if abs(raw_adjustment) < 50:

        actual_theory = 0

        carry = raw_adjustment

    else:

        actual_theory = raw_adjustment

        carry = 0

    result.append({

        "date": current_date,

        "price": current_avg,

        "oil_change": delta_price,

        "theoretical_adjustment":
            actual_theory,

        "actual_adjustment":
            adj.loc[i,
            "actual_adjustment"]
    })

# ======================
# 5. save result
# ======================

result_df = pd.DataFrame(result)

result_df.to_csv(

    "merged_result.csv",

    index=False
)

print("\n===== mechanism model complete =====\n")

print(result_df.head())