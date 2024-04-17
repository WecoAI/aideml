import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Prepare the data
X = train_data.drop(["id", "target"], axis=1)
y = train_data["target"]
X_test = test_data.drop("id", axis=1)

# Prepare cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = []

# Perform cross-validation
for train_index, valid_index in kf.split(X):
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # Create LightGBM datasets
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    # Train the model
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }
    model = lgb.train(params, lgb_train, valid_sets=[lgb_valid], verbose_eval=False)

    # Predict on validation set
    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)

    # Evaluate the model
    auc_score = roc_auc_score(y_valid, y_pred)
    auc_scores.append(auc_score)

# Calculate the average AUC score
average_auc_score = sum(auc_scores) / len(auc_scores)
print(f"Average AUC score from cross-validation: {average_auc_score}")

# Train the model on the full dataset
full_train_set = lgb.Dataset(X, y)
final_model = lgb.train(params, full_train_set)

# Predict on the test set
predictions = final_model.predict(X_test, num_iteration=final_model.best_iteration)

# Prepare the submission file
submission = pd.DataFrame({"id": test_data["id"], "target": predictions})
submission.to_csv("./working/submission.csv", index=False)
