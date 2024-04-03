import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFECV
import lightgbm as lgb
from bayes_opt import BayesianOptimization

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Prepare the data
X = train_data.drop(["id", "defects"], axis=1)
y = train_data["defects"]
X_test = test_data.drop("id", axis=1)
test_ids = test_data["id"]

# Initialize LightGBM model with the best parameters from previous optimization
best_params = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_depth": 15,
    "reg_alpha": 0.5,
    "reg_lambda": 0.5,
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
}
lgb_model = lgb.LGBMClassifier(**best_params)

# Perform feature selection using RFECV
rfecv = RFECV(estimator=lgb_model, step=1, cv=KFold(10), scoring="roc_auc", n_jobs=-1)
rfecv.fit(X, y)

# Print the optimal number of features
print(f"Optimal number of features: {rfecv.n_features_}")

# Select the optimal features
X_selected = rfecv.transform(X)
X_test_selected = rfecv.transform(X_test)

# Retrain the model with the selected features
lgb_model.fit(X_selected, y)

# Predict on the test set with the selected features
final_predictions = lgb_model.predict_proba(X_test_selected)[:, 1]

# Save the submission file
submission = pd.DataFrame({"id": test_ids, "defects": final_predictions})
submission.to_csv("./working/submission.csv", index=False)

# Evaluate the model with selected features using cross-validation
auc_scores = []
kf = KFold(n_splits=10, shuffle=True, random_state=42)
for train_index, valid_index in kf.split(X_selected):
    X_train, X_valid = X_selected[train_index], X_selected[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    lgb_model.fit(X_train, y_train)
    y_pred = lgb_model.predict_proba(X_valid)[:, 1]
    auc_score = roc_auc_score(y_valid, y_pred)
    auc_scores.append(auc_score)

# Print the mean AUC score
mean_auc_score = np.mean(auc_scores)
print(f"Mean AUC Score with Selected Features: {mean_auc_score}")
