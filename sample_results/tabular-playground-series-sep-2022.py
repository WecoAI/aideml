import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")


# Feature engineering
def create_features(df):
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["date"].dt.weekday >= 5).astype(int)
    return df


train_data = create_features(train_data)
test_data = create_features(test_data)

# One-hot encoding for categorical features
ohe = OneHotEncoder(sparse=False)
ohe.fit(train_data[["country", "store", "product"]])
train_encoded = ohe.transform(train_data[["country", "store", "product"]])
test_encoded = ohe.transform(test_data[["country", "store", "product"]])

# Prepare the final train and test sets
X_train = np.hstack(
    (train_data[["year", "month", "day", "dayofweek", "is_weekend"]], train_encoded)
)
y_train = train_data["num_sold"]
X_test = np.hstack(
    (test_data[["year", "month", "day", "dayofweek", "is_weekend"]], test_encoded)
)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Train the model
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# Make predictions on the validation set
val_predictions = model.predict(X_val)


# Calculate SMAPE
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)


smape_score = smape(y_val, val_predictions)
print(f"SMAPE: {smape_score}")

# Retrain the model on the full training set and make predictions on the test set
model.fit(X_train, y_train)
test_predictions = model.predict(X_test)

# Save the predictions to a CSV file
submission = pd.DataFrame({"row_id": test_data["row_id"], "num_sold": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
