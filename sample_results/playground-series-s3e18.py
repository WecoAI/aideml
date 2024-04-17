import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Identify common features
common_features = list(set(train_data.columns) & set(test_data.columns))
common_features.remove("id")

# Prepare the data
X_train = train_data[common_features]
y_train_EC1 = train_data["EC1"]
y_train_EC2 = train_data["EC2"]
X_test = test_data[common_features]

# Initialize StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize individual models
model1_EC1 = LGBMClassifier(random_state=42)
model1_EC2 = LGBMClassifier(random_state=42)
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model3 = LogisticRegression(max_iter=1000)
model4 = GradientBoostingClassifier(random_state=42)

# Define the parameter grid for GradientBoostingClassifier
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
}

# Perform GridSearchCV to find the best parameters for GradientBoostingClassifier
grid_search = GridSearchCV(model4, param_grid, cv=skf, scoring="roc_auc")
grid_search.fit(
    X_train, y_train_EC1
)  # We can use y_train_EC1 to find general good params
best_params = grid_search.best_params_

# Update the GradientBoostingClassifier with the best parameters
model4 = GradientBoostingClassifier(random_state=42, **best_params)

# Combine models into a VotingClassifier with soft voting
voting_clf_EC1 = VotingClassifier(
    estimators=[("lgbm", model1_EC1), ("rf", model2), ("lr", model3), ("gbc", model4)],
    voting="soft",
)
voting_clf_EC2 = VotingClassifier(
    estimators=[("lgbm", model1_EC2), ("rf", model2), ("lr", model3), ("gbc", model4)],
    voting="soft",
)

# Train and evaluate the ensemble model for EC1
cv_scores_EC1 = cross_val_score(
    voting_clf_EC1, X_train, y_train_EC1, cv=skf, scoring="roc_auc"
)
auc_EC1 = np.mean(cv_scores_EC1)

# Train and evaluate the ensemble model for EC2
cv_scores_EC2 = cross_val_score(
    voting_clf_EC2, X_train, y_train_EC2, cv=skf, scoring="roc_auc"
)
auc_EC2 = np.mean(cv_scores_EC2)

# Print the evaluation metric for each target
print(f"Validation AUC for EC1: {auc_EC1}")
print(f"Validation AUC for EC2: {auc_EC2}")
print(f"Average Validation AUC: {(auc_EC1 + auc_EC2) / 2}")

# Fit the ensemble models on the entire training set
voting_clf_EC1.fit(X_train, y_train_EC1)
voting_clf_EC2.fit(X_train, y_train_EC2)

# Predict probabilities for the test set
test_data["EC1"] = voting_clf_EC1.predict_proba(X_test)[:, 1]
test_data["EC2"] = voting_clf_EC2.predict_proba(X_test)[:, 1]

# Prepare the submission file
submission = test_data[["id", "EC1", "EC2"]]
submission.to_csv("./working/submission.csv", index=False)
