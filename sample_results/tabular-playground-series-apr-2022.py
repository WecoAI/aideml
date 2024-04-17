import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Load the data
train_data = pd.read_csv("./input/train.csv")
train_labels = pd.read_csv("./input/train_labels.csv")
test_data = pd.read_csv("./input/test.csv")

# Aggregate features for each sequence
agg_funcs = ["mean", "std", "min", "max"]
train_features = train_data.groupby("sequence").agg(agg_funcs)
test_features = test_data.groupby("sequence").agg(agg_funcs)

# Flatten multi-level columns
train_features.columns = [
    "_".join(col).strip() for col in train_features.columns.values
]
test_features.columns = ["_".join(col).strip() for col in test_features.columns.values]

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_features, train_labels["state"], test_size=0.2, random_state=42
)

# Initialize and train the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict probabilities for the validation set
val_probs = rf.predict_proba(X_val)[:, 1]

# Calculate the AUC-ROC score
auc_score = roc_auc_score(y_val, val_probs)
print(f"AUC-ROC score: {auc_score}")

# Predict probabilities for the test set
test_probs = rf.predict_proba(test_features)[:, 1]

# Create the submission file
submission = pd.DataFrame({"sequence": test_features.index, "state": test_probs})
submission.to_csv("./working/submission.csv", index=False)
