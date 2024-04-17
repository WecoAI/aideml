import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Load the data
train_data = pd.read_csv("./input/train.csv")
X_train = train_data.drop(["id", "target"], axis=1)
y_train = train_data["target"]

# Initialize the model with L1 regularization
model = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)

# Prepare cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = []

# Perform 10-fold cross-validation
for train_idx, valid_idx in cv.split(X_train, y_train):
    X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]

    # Train the model
    model.fit(X_train_fold, y_train_fold)

    # Predict probabilities for the validation set
    y_pred_prob = model.predict_proba(X_valid_fold)[:, 1]

    # Calculate the AUC score and append to the list
    auc_score = roc_auc_score(y_valid_fold, y_pred_prob)
    auc_scores.append(auc_score)

# Calculate the average AUC score across all folds
average_auc_score = sum(auc_scores) / len(auc_scores)
print(f"Average AUC-ROC score: {average_auc_score}")

# Train the model on the full training set and predict for the test set
model.fit(X_train, y_train)
test_data = pd.read_csv("./input/test.csv")
X_test = test_data.drop("id", axis=1)
test_data["target"] = model.predict_proba(X_test)[:, 1]

# Save the submission file
submission_file = "./working/submission.csv"
test_data[["id", "target"]].to_csv(submission_file, index=False)
