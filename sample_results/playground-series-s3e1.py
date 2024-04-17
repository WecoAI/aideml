import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")

# Prepare the data
X = train_data.drop(["MedHouseVal", "id"], axis=1)
y = train_data["MedHouseVal"]
X_test = test_data.drop("id", axis=1)

# Prepare cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
rmse_scores = []

# Perform 10-fold cross-validation
for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Create LightGBM datasets
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val)

    # Train the model
    params = {"objective": "regression", "metric": "rmse", "verbosity": -1}
    model = lgb.train(
        params, train_set, valid_sets=[train_set, val_set], verbose_eval=False
    )

    # Predict on validation set
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_scores.append(rmse)

# Print the average RMSE across the folds
print(f"Average RMSE: {np.mean(rmse_scores)}")

# Train the model on the full dataset
full_train_set = lgb.Dataset(X, label=y)
final_model = lgb.train(params, full_train_set, verbose_eval=False)

# Predict on the test set
predictions = final_model.predict(X_test, num_iteration=final_model.best_iteration)

# Prepare the submission file
submission = pd.DataFrame({"id": test_data["id"], "MedHouseVal": predictions})
submission.to_csv("./working/submission.csv", index=False)
