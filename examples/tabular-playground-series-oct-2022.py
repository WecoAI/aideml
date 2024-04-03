import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Read dtypes and replace 'float16' with 'float32'
dtypes_df = pd.read_csv("./input/train_dtypes.csv")
dtypes = {
    k: (v if v != "float16" else "float32")
    for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)
}

# Read and concatenate training data
train_dfs = [pd.read_csv(f"./input/train_{i}.csv", dtype=dtypes) for i in range(10)]
train_df = pd.concat(train_dfs, ignore_index=True)

# Prepare the data
X = train_df.drop(
    [
        "game_num",
        "event_id",
        "event_time",
        "player_scoring_next",
        "team_scoring_next",
        "team_A_scoring_within_10sec",
        "team_B_scoring_within_10sec",
    ],
    axis=1,
)
y = train_df[["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]]

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model for team A
model_A = lgb.LGBMClassifier()
model_A.fit(X_train, y_train.iloc[:, 0])

# Train the model for team B
model_B = lgb.LGBMClassifier()
model_B.fit(X_train, y_train.iloc[:, 1])

# Predict on validation set for team A
val_preds_A = model_A.predict_proba(X_val)[:, 1]

# Predict on validation set for team B
val_preds_B = model_B.predict_proba(X_val)[:, 1]

# Combine predictions
val_preds = pd.DataFrame(
    {
        "team_A_scoring_within_10sec": val_preds_A,
        "team_B_scoring_within_10sec": val_preds_B,
    }
)

# Calculate log loss
val_log_loss = log_loss(y_val, val_preds)
print(f"Validation Log Loss: {val_log_loss}")

# Predict on test set
test_dtypes_df = pd.read_csv("./input/test_dtypes.csv")
test_dtypes = {
    k: (v if v != "float16" else "float32")
    for (k, v) in zip(test_dtypes_df.column, test_dtypes_df.dtype)
}
test_df = pd.read_csv("./input/test.csv", dtype=test_dtypes)
X_test = test_df.drop(["id"], axis=1)

# Predict on test set for team A
test_preds_A = model_A.predict_proba(X_test)[:, 1]

# Predict on test set for team B
test_preds_B = model_B.predict_proba(X_test)[:, 1]

# Combine predictions
test_preds = pd.DataFrame(
    {
        "team_A_scoring_within_10sec": test_preds_A,
        "team_B_scoring_within_10sec": test_preds_B,
    }
)

# Prepare submission
submission = pd.concat([test_df["id"], test_preds], axis=1)
submission.to_csv("./working/submission.csv", index=False)
