import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Preprocess the data
train_data[["ID", "latitude", "longitude", "year", "week_no"]] = train_data[
    "ID_LAT_LON_YEAR_WEEK"
].str.split("_", expand=True)
test_data[["ID", "latitude", "longitude", "year", "week_no"]] = test_data[
    "ID_LAT_LON_YEAR_WEEK"
].str.split("_", expand=True)

# Convert to numeric types
for col in ["latitude", "longitude", "year", "week_no"]:
    train_data[col] = pd.to_numeric(train_data[col])
    test_data[col] = pd.to_numeric(test_data[col])

# One-hot encoding for 'week_no'
train_data = pd.get_dummies(train_data, columns=["week_no"])
test_data = pd.get_dummies(test_data, columns=["week_no"])

# Align test_data columns with train_data
test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

# Prepare the data for training
X = train_data.drop(columns=["emission", "ID_LAT_LON_YEAR_WEEK", "ID"])
y = train_data["emission"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the LightGBM model
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

params = {"objective": "regression", "metric": "rmse", "verbose": -1}

gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=100,
    valid_sets=lgb_eval,
    early_stopping_rounds=10,
)

# Predict on validation set
y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)

# Evaluate the model
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f"Validation RMSE: {rmse}")

# Predict on test set and save submission
test_features = test_data.drop(columns=["ID_LAT_LON_YEAR_WEEK", "ID", "emission"])
test_data["emission"] = gbm.predict(test_features, num_iteration=gbm.best_iteration)
submission = test_data[["ID_LAT_LON_YEAR_WEEK", "emission"]]
submission.to_csv("./working/submission.csv", index=False)
