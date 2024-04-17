import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

# Load data
sales = pd.read_csv("./input/sales_train.csv")
test = pd.read_csv("./input/test.csv")

# Convert date to datetime and extract year and month
sales["date"] = pd.to_datetime(sales["date"], format="%d.%m.%Y")
sales["year"] = sales["date"].dt.year
sales["month"] = sales["date"].dt.month

# Aggregate data to monthly level
monthly_sales = (
    sales.groupby(["year", "month", "shop_id", "item_id"])
    .agg({"item_cnt_day": "sum"})
    .reset_index()
)
monthly_sales.rename(columns={"item_cnt_day": "item_cnt_month"}, inplace=True)

# Create lag features
for lag in [1, 2, 3]:
    shifted = monthly_sales.copy()
    shifted["month"] += lag
    shifted["year"] += shifted["month"] // 12
    shifted["month"] %= 12
    shifted.rename(
        columns={"item_cnt_month": f"item_cnt_month_lag_{lag}"}, inplace=True
    )
    monthly_sales = pd.merge(
        monthly_sales, shifted, on=["year", "month", "shop_id", "item_id"], how="left"
    )

# Mean encoded features
item_mean = monthly_sales.groupby("item_id")["item_cnt_month"].mean().reset_index()
item_mean.rename(columns={"item_cnt_month": "item_mean_cnt"}, inplace=True)
shop_mean = monthly_sales.groupby("shop_id")["item_cnt_month"].mean().reset_index()
shop_mean.rename(columns={"item_cnt_month": "shop_mean_cnt"}, inplace=True)

monthly_sales = pd.merge(monthly_sales, item_mean, on="item_id", how="left")
monthly_sales = pd.merge(monthly_sales, shop_mean, on="shop_id", how="left")

# Prepare training data
X = monthly_sales.drop(["item_cnt_month", "year", "month"], axis=1)
y = monthly_sales["item_cnt_month"].clip(0, 20)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LGBMRegressor()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_val).clip(0, 20)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse}")

# Prepare test set
test = pd.merge(
    test,
    monthly_sales.drop(["item_cnt_month"], axis=1),
    on=["shop_id", "item_id"],
    how="left",
).fillna(0)

# Drop 'year' and 'month' columns to match training data
test.drop(["year", "month"], axis=1, inplace=True)

# Make predictions on test set
test["item_cnt_month"] = model.predict(test.drop(["ID"], axis=1)).clip(0, 20)

# Save submission
test[["ID", "item_cnt_month"]].to_csv("./working/submission.csv", index=False)
