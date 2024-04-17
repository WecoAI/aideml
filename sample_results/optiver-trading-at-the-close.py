import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Preprocess the data: fill missing values with median for numeric columns only
numeric_columns_train = train_data.select_dtypes(include=[np.number]).columns
train_data[numeric_columns_train] = train_data[numeric_columns_train].fillna(
    train_data[numeric_columns_train].median()
)

# Ensure 'target' is not in the numeric columns for test data
numeric_columns_test = test_data.select_dtypes(include=[np.number]).columns
test_data[numeric_columns_test] = test_data[numeric_columns_test].fillna(
    test_data[numeric_columns_test].median()
)

# Prepare features and target
X = train_data.drop(["row_id", "target"], axis=1)
y = train_data["target"]

# Initialize LightGBM regressor
model = LGBMRegressor()

# Prepare cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
mae_scores = []

# Perform 10-fold cross-validation
for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Train the model
    model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = model.predict(X_val)

    # Calculate and store MAE
    mae = mean_absolute_error(y_val, y_pred)
    mae_scores.append(mae)

# Print the average MAE across all folds
print(f"Average MAE: {np.mean(mae_scores)}")

# Predict on test set
test_features = test_data.drop(["row_id"], axis=1)
test_data["target"] = model.predict(test_features)

# Save predictions to submission.csv
submission = test_data[["row_id", "target"]]
submission.to_csv("./working/submission.csv", index=False)
