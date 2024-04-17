import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Prepare the data
X = train_data.drop(["id", "booking_status"], axis=1)
y = train_data["booking_status"]
X_test = test_data.drop("id", axis=1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict on the validation set
val_predictions = model.predict_proba(X_val)[:, 1]
val_roc_auc = roc_auc_score(y_val, val_predictions)
print(f"Validation ROC AUC: {val_roc_auc}")

# Train the model on the full training data and predict on the test set
model.fit(X, y)
test_predictions = model.predict_proba(X_test)[:, 1]

# Save the predictions in the submission format
submission = pd.DataFrame({"id": test_data["id"], "booking_status": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
