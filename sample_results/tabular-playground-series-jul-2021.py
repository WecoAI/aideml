import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_log_error

# Load the datasets
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Define the target variables explicitly
targets = ["target_carbon_monoxide", "target_benzene", "target_nitrogen_oxides"]

# Feature engineering: Convert date_time to datetime and extract useful features
train["date_time"] = pd.to_datetime(train["date_time"])
test["date_time"] = pd.to_datetime(test["date_time"])

# Extracting datetime features
for df in [train, test]:
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["day_of_month"] = df["date_time"].dt.day
    df["month"] = df["date_time"].dt.month

# Creating interaction terms between sensor readings and weather features
for sensor in ["sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5"]:
    for weather in ["deg_C", "relative_humidity", "absolute_humidity"]:
        train[f"{sensor}_{weather}_interaction"] = train[sensor] * train[weather]
        test[f"{sensor}_{weather}_interaction"] = test[sensor] * test[weather]

# Update features list to include the new interaction terms
features = train.columns.drop(["date_time"] + targets).tolist()

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train[features], train[targets], test_size=0.2, random_state=42
)

# Hyperparameter tuning setup
param_grid = {
    "num_leaves": [31, 50, 70],
    "max_depth": [-1, 10, 20],
    "learning_rate": [0.1, 0.01, 0.05],
    "n_estimators": [100, 200, 500],
}

# Retrain models using only the selected features and hyperparameter tuning
rmsle_scores = []
for target in targets:
    model = LGBMRegressor()
    random_search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=10,
        scoring="neg_mean_squared_log_error",
        cv=3,
        random_state=42,
    )
    random_search.fit(X_train, y_train[target])
    best_model = random_search.best_estimator_
    predictions = best_model.predict(X_val)
    rmsle_score = np.sqrt(mean_squared_log_error(y_val[target], predictions))
    rmsle_scores.append(rmsle_score)

# Calculate and print the mean RMSLE score
mean_rmsle = np.mean(rmsle_scores)
print(f"Mean RMSLE after hyperparameter tuning: {mean_rmsle}")

# Prepare submission using hyperparameter tuned models
test_predictions = pd.DataFrame({"date_time": test["date_time"]})
for target in targets:
    model = LGBMRegressor()
    random_search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=10,
        scoring="neg_mean_squared_log_error",
        cv=3,
        random_state=42,
    )
    random_search.fit(train[features], train[target])
    best_model = random_search.best_estimator_
    test_predictions[target] = best_model.predict(test[features])

# Renaming columns as required for submission
test_predictions.rename(
    columns={
        "target_carbon_monoxide": "target_carbon_monoxide",
        "target_benzene": "target_benzene",
        "target_nitrogen_oxides": "target_nitrogen_oxides",
    },
    inplace=True,
)

# Saving the submission file
test_predictions.to_csv("./working/submission.csv", index=False)
