import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Separate features and target
X = train_data.drop(["id", "target"], axis=1)
y = train_data["target"]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical features
cat_features = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Define hyperparameter space
learning_rates = [0.03, 0.1]
depths = [4, 6, 8]
n_estimators = [100, 500, 1000]

best_rmse = float("inf")
best_params = {}

# Grid search
for learning_rate in learning_rates:
    for depth in depths:
        for n_estimator in n_estimators:
            model = CatBoostRegressor(
                loss_function="RMSE",
                cat_features=cat_features,
                verbose=200,
                random_seed=42,
                learning_rate=learning_rate,
                depth=depth,
                n_estimators=n_estimator,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                use_best_model=True,
            )
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {
                    "learning_rate": learning_rate,
                    "depth": depth,
                    "n_estimators": n_estimator,
                }

# Train the model with best parameters
model = CatBoostRegressor(
    loss_function="RMSE",
    cat_features=cat_features,
    verbose=200,
    random_seed=42,
    **best_params,
)
model.fit(X, y)

# Predict on test data
test_predictions = model.predict(test_data.drop(["id"], axis=1))

# Save test predictions to file
submission = pd.DataFrame({"id": test_data["id"], "target": test_predictions})
submission.to_csv("./working/submission.csv", index=False)

print(f"Best Validation RMSE: {best_rmse}")
print(f"Best Parameters: {best_params}")
