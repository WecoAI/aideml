import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")

# Prepare the data
X = train_data.drop(["id", "target"], axis=1)
y = train_data["target"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Prepare the test set
X_test = test_data.drop(["id"], axis=1)

# Train the model
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

params = {
    "objective": "multiclass",
    "num_class": 9,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": -1,
    "verbose": -1,
}

model = lgb.train(
    params,
    train_set,
    valid_sets=[train_set, val_set],
    early_stopping_rounds=100,
    num_boost_round=1000,
    verbose_eval=100,
)

# Evaluate the model
val_predictions = model.predict(X_val)
val_log_loss = log_loss(y_val, val_predictions)
print(f"Validation Log Loss: {val_log_loss}")

# Make predictions on the test set
test_predictions = model.predict(X_test)

# Prepare submission file
submission = pd.DataFrame(
    test_predictions, columns=["Class_" + str(i + 1) for i in range(9)]
)
submission["id"] = test_data["id"]
submission = submission[["id"] + ["Class_" + str(i + 1) for i in range(9)]]
submission.to_csv("./working/submission.csv", index=False)
