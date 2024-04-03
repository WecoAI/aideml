import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error

# Load the data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")


# Feature engineering
def preprocess_data(data):
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["hour"] = data["datetime"].dt.hour
    data["day_of_week"] = data["datetime"].dt.dayofweek
    data["month"] = data["datetime"].dt.month
    data["year"] = data["datetime"].dt.year
    data["day"] = data["datetime"].dt.day
    data["hour_workingday_interaction"] = data["hour"] * data["workingday"]

    # Adding cyclic features
    data["hour_sin"] = np.sin(data.hour * (2.0 * np.pi / 24))
    data["hour_cos"] = np.cos(data.hour * (2.0 * np.pi / 24))
    data["day_of_week_sin"] = np.sin(data.day_of_week * (2.0 * np.pi / 7))
    data["day_of_week_cos"] = np.cos(data.day_of_week * (2.0 * np.pi / 7))
    data["month_sin"] = np.sin((data.month - 1) * (2.0 * np.pi / 12))
    data["month_cos"] = np.cos((data.month - 1) * (2.0 * np.pi / 12))

    return data.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")


train = preprocess_data(train)
test = preprocess_data(test)

# Splitting the training data for validation
X = train.drop(["count"], axis=1)
y = np.log1p(train["count"])  # Apply log1p to transform the target variable
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_val)
rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(y_pred)))
print(f"RMSLE with cyclic features: {rmsle}")

# Prepare submission
test_pred = model.predict(test)
submission = pd.DataFrame(
    {
        "datetime": pd.read_csv("./input/test.csv")["datetime"],
        "count": np.expm1(test_pred),
    }
)
submission.to_csv("./working/submission.csv", index=False)
