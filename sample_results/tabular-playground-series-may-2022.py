import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Frequency encode the 'f_27' feature
freq_encoder = train_data["f_27"].value_counts(normalize=True)
train_data["f_27"] = train_data["f_27"].map(freq_encoder)
test_data["f_27"] = test_data["f_27"].map(freq_encoder).fillna(0)

# Separate features and target
X = train_data.drop(["id", "target"], axis=1)
y = train_data["target"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LGBMClassifier()

# Train the model
model.fit(X_train, y_train)

# Predict probabilities for the validation set
val_probs = model.predict_proba(X_val)[:, 1]

# Calculate the ROC AUC score
val_auc = roc_auc_score(y_val, val_probs)
print(f"Validation ROC AUC Score: {val_auc}")

# Predict probabilities for the test set
test_probs = model.predict_proba(test_data.drop(["id"], axis=1))[:, 1]

# Create a submission file
submission = pd.DataFrame({"id": test_data["id"], "target": test_probs})
submission.to_csv("./working/submission.csv", index=False)
