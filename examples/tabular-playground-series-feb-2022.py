import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Encode the 'target' column
le = LabelEncoder()
train_data["target"] = le.fit_transform(train_data["target"])

# Separate features and target
X = train_data.drop(["row_id", "target"], axis=1)
y = train_data["target"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM datasets
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val)

# Adjusted parameters for LightGBM
params = {
    "objective": "multiclass",
    "num_class": len(le.classes_),
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 50,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.75,  # Adjusted feature fraction
    "bagging_fraction": 0.85,  # Adjusted bagging fraction
    "bagging_freq": 5,
}

# Train the model
gbm = lgb.train(
    params,
    train_set,
    num_boost_round=1000,
    valid_sets=[train_set, val_set],
    early_stopping_rounds=100,
    verbose_eval=100,
)

# Predict on validation set
y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
y_pred_max = [np.argmax(line) for line in y_pred]

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred_max)
print(f"Validation Accuracy: {accuracy}")

# Predict on test set
X_test = test_data.drop(["row_id"], axis=1)
test_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
test_pred_max = [np.argmax(line) for line in test_pred]

# Inverse transform the predicted labels
test_pred_labels = le.inverse_transform(test_pred_max)

# Prepare submission
submission = pd.DataFrame({"row_id": test_data["row_id"], "target": test_pred_labels})
submission.to_csv("./working/submission.csv", index=False)
