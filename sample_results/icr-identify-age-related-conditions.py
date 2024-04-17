import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import log_loss, make_scorer
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Encode categorical features
le = LabelEncoder()
train_data["EJ"] = le.fit_transform(train_data["EJ"])
test_data["EJ"] = le.transform(test_data["EJ"])

# Prepare the data
X = train_data.drop(["Id", "Class"], axis=1)
y = train_data["Class"]
X_test = test_data.drop("Id", axis=1)

# Define the model parameters and parameter grid for randomized search
model = lgb.LGBMClassifier(objective="binary", boosting_type="gbdt", is_unbalance=True)
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [15, 31, 63],
    "max_depth": [-1, 5, 10],
    "min_child_samples": [10, 20, 30],
    "max_bin": [255, 300],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.3, 0.5, 0.7],
}

# Create a scorer for log loss
log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

# Perform randomized search with cross-validation
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_grid,
    n_iter=10,
    scoring=log_loss_scorer,
    cv=KFold(n_splits=10, shuffle=True, random_state=42),
    random_state=42,
    verbose=1,
)

random_search.fit(X, y)

# Best model and log loss
best_model = random_search.best_estimator_
best_score = -random_search.best_score_
print(f"Best Log Loss: {best_score}")

# Predict on test set with the best model
test_predictions = best_model.predict_proba(X_test)[:, 1]

# Create a submission file
submission = pd.DataFrame(
    {
        "Id": test_data["Id"],
        "class_0": 1 - test_predictions,
        "class_1": test_predictions,
    }
)
submission.to_csv("./working/submission.csv", index=False)
