import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Prepare the data
X = train_data.drop(["id", "smoking"], axis=1)
y = train_data["smoking"]
X_test = test_data.drop("id", axis=1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Define the LightGBM cross-validation function
def lgb_cv(
    learning_rate,
    num_leaves,
    min_child_samples,
    subsample,
    colsample_bytree,
    max_depth,
    reg_alpha,
    reg_lambda,
    n_estimators,
):
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": max(min(learning_rate, 1), 0),
        "n_estimators": int(n_estimators),
        "verbose": -1,
        "num_leaves": int(num_leaves),
        "min_child_samples": int(min_child_samples),
        "subsample": max(min(subsample, 1), 0),
        "colsample_bytree": max(min(colsample_bytree, 1), 0),
        "max_depth": int(max_depth),
        "reg_alpha": max(reg_alpha, 0),
        "reg_lambda": max(reg_lambda, 0),
    }
    cv_result = lgb.cv(
        params,
        lgb.Dataset(X_train_scaled, label=y_train),
        nfold=10,
        seed=42,
        stratified=True,
        verbose_eval=200,
        metrics=["auc"],
    )
    return max(cv_result["auc-mean"])


# Define the parameter bounds
param_bounds = {
    "learning_rate": (0.01, 0.2),
    "num_leaves": (20, 60),
    "min_child_samples": (5, 50),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "max_depth": (5, 15),
    "reg_alpha": (0, 1),
    "reg_lambda": (0, 1),
    "n_estimators": (100, 1000),  # Increased range for n_estimators
}

# Perform Bayesian optimization with increased initial points and iterations
optimizer = BayesianOptimization(f=lgb_cv, pbounds=param_bounds, random_state=42)
optimizer.maximize(init_points=10, n_iter=50)

# Retrieve the best parameters
best_params = optimizer.max["params"]
best_params["num_leaves"] = int(best_params["num_leaves"])
best_params["min_child_samples"] = int(best_params["min_child_samples"])
best_params["max_depth"] = int(best_params["max_depth"])
best_params["n_estimators"] = int(best_params["n_estimators"])

# Train and validate the model with the best parameters
final_gbm = lgb.LGBMClassifier(**best_params)
final_gbm.fit(X_train_scaled, y_train)
val_predictions = final_gbm.predict_proba(X_val_scaled)[:, 1]
val_auc = roc_auc_score(y_val, val_predictions)
print(f"Validation AUC score: {val_auc}")

# Train the model on the full dataset with the best parameters and make predictions on the scaled test set
final_gbm.fit(scaler.fit_transform(X), y)
predictions = final_gbm.predict_proba(X_test_scaled)[:, 1]

# Prepare the submission file
submission = pd.DataFrame({"id": test_data["id"], "smoking": predictions})
submission.to_csv("./working/submission.csv", index=False)
