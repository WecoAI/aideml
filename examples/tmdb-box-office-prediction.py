import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder

# Load the data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Feature engineering on 'release_date'
train_df["release_date"] = pd.to_datetime(train_df["release_date"])
test_df["release_date"] = pd.to_datetime(test_df["release_date"], errors="coerce")

train_df["release_year"] = train_df["release_date"].dt.year
train_df["release_month"] = train_df["release_date"].dt.month
train_df["release_dayofweek"] = train_df["release_date"].dt.dayofweek

test_df["release_year"] = test_df["release_date"].dt.year
test_df["release_month"] = test_df["release_date"].dt.month
test_df["release_dayofweek"] = test_df["release_date"].dt.dayofweek

# Fill missing 'release_date' derived features in test set with median from train set
test_df["release_year"] = test_df["release_year"].fillna(
    train_df["release_year"].median()
)
test_df["release_month"] = test_df["release_month"].fillna(
    train_df["release_month"].median()
)
test_df["release_dayofweek"] = test_df["release_dayofweek"].fillna(
    train_df["release_dayofweek"].median()
)

# Select features for the model
features = [
    "budget",
    "popularity",
    "runtime",
    "original_language",
    "release_year",
    "release_month",
    "release_dayofweek",
]
target = "revenue"

# Process categorical features with one-hot encoding
categorical_features = ["original_language"]
encoder = OneHotEncoder(handle_unknown="ignore")
encoded_features = encoder.fit_transform(train_df[categorical_features]).toarray()
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

# Fill missing numerical feature values with median
for feature in features:
    if feature not in categorical_features:
        train_df[feature] = train_df[feature].fillna(train_df[feature].median())

# Combine numerical and encoded categorical features
X = pd.concat(
    [train_df[features].drop(columns=categorical_features), encoded_df], axis=1
)
y = np.log1p(train_df[target])

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the LightGBM model
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_valid)

# Calculate the RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_valid, y_pred))
print(f"RMSLE: {rmsle}")

# Prepare the test set
test_encoded_features = encoder.transform(test_df[categorical_features]).toarray()
test_encoded_df = pd.DataFrame(test_encoded_features, columns=encoded_feature_names)
# Fill missing numerical feature values with median in the test set
for feature in features:
    if feature not in categorical_features:
        test_df[feature] = test_df[feature].fillna(train_df[feature].median())
X_test = pd.concat(
    [test_df[features].drop(columns=categorical_features), test_encoded_df], axis=1
)

# Predict on the test set
test_pred = model.predict(X_test)

# Prepare the submission file
submission = pd.DataFrame({"id": test_df["id"], "revenue": np.expm1(test_pred)})
submission.to_csv("./working/submission.csv", index=False)
