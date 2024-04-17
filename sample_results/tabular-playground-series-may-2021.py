import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Prepare the data
X = train_data.drop(["id", "target"], axis=1)
y = train_data["target"].astype("category")
X_test = test_data.drop(["id"], axis=1)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = lgb.LGBMClassifier(objective="multiclass", random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
val_preds = model.predict_proba(X_val)
print(f"Validation Log Loss: {log_loss(y_val, val_preds)}")

# Predict on test set
test_preds = model.predict_proba(X_test)

# Prepare the submission file
submission = pd.DataFrame(
    test_preds, columns=["Class_1", "Class_2", "Class_3", "Class_4"]
)
submission["id"] = test_data["id"]
submission = submission[["id", "Class_1", "Class_2", "Class_3", "Class_4"]]
submission.to_csv("./working/submission.csv", index=False)
