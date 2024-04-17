import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Separate features and target
X = train_data.drop(columns=["id", "claim"])
y = train_data["claim"]

# Handle missing values by imputing with median
X.fillna(X.median(), inplace=True)
test_data.fillna(test_data.median(), inplace=True)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LightGBM model
model = LGBMClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the validation set
val_predictions = model.predict_proba(X_val)[:, 1]

# Calculate the ROC AUC score
val_auc = roc_auc_score(y_val, val_predictions)
print(f"Validation ROC AUC Score: {val_auc}")

# Predict on the test set
test_predictions = model.predict_proba(test_data.drop(columns=["id"]))[:, 1]

# Create the submission file
submission = pd.DataFrame({"id": test_data["id"], "claim": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
