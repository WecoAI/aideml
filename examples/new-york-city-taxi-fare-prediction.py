import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Load a subset of the training data
train_df = pd.read_csv("./input/train.csv", nrows=500000)

# Remove missing values and outliers
train_df = train_df.dropna(how="any", axis="rows")
train_df = train_df[(train_df.fare_amount >= 2.5) & (train_df.fare_amount <= 500)]
train_df = train_df[(train_df.passenger_count > 0) & (train_df.passenger_count <= 6)]
train_df = train_df[
    (train_df["pickup_latitude"] != 0) | (train_df["pickup_longitude"] != 0)
]
train_df = train_df[
    (train_df["dropoff_latitude"] != 0) | (train_df["dropoff_longitude"] != 0)
]


# Feature engineering
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # radius of Earth in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return d


train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
train_df["year"] = train_df["pickup_datetime"].dt.year
train_df["month"] = train_df["pickup_datetime"].dt.month
train_df["day"] = train_df["pickup_datetime"].dt.day
train_df["hour"] = train_df["pickup_datetime"].dt.hour
train_df["weekday"] = train_df["pickup_datetime"].dt.weekday
train_df["distance"] = haversine_distance(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"],
)

# Select features and target variable
features = [
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "passenger_count",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "distance",
]
target = "fare_amount"

X = train_df[features]
y = train_df[target]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf = RandomForestRegressor(n_estimators=50, max_depth=25, random_state=42)
rf.fit(X_train, y_train)

# Predict on validation set
y_pred = rf.predict(X_val)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse}")

# Prepare the test set
test_df = pd.read_csv("./input/test.csv")
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])
test_df["year"] = test_df["pickup_datetime"].dt.year
test_df["month"] = test_df["pickup_datetime"].dt.month
test_df["day"] = test_df["pickup_datetime"].dt.day
test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["weekday"] = test_df["pickup_datetime"].dt.weekday

# Impute NaN values in the test set using median from the training set
for feature in [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]:
    median_value = train_df[feature].median()
    test_df[feature].fillna(median_value, inplace=True)

test_df["distance"] = haversine_distance(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
)

# Predict on test set
X_test = test_df[features]
test_df["fare_amount"] = rf.predict(X_test)

# Save predictions
submission = test_df[["key", "fare_amount"]]
submission.to_csv("./working/submission.csv", index=False)
