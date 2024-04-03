import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Separate target from predictors
y = train_data["target"]
X = train_data.drop(["target", "id"], axis=1)
X_test = test_data.drop("id", axis=1)

# List of columns by type
binary_cols = [col for col in X.columns if "bin" in col]
ordinal_cols = [col for col in X.columns if "ord" in col]
nominal_cols = [col for col in X.columns if "nom" in col]
cyclical_cols = ["day", "month"]

# Ordinal encoding for binary and ordinal features
ordinal_encoder = OrdinalEncoder()
X[binary_cols + ordinal_cols] = ordinal_encoder.fit_transform(
    X[binary_cols + ordinal_cols]
)
X_test[binary_cols + ordinal_cols] = ordinal_encoder.transform(
    X_test[binary_cols + ordinal_cols]
)

# One-hot encoding for nominal features with low cardinality
low_cardinality_nom_cols = [col for col in nominal_cols if X[col].nunique() < 10]
one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
X_low_card_nom = pd.DataFrame(
    one_hot_encoder.fit_transform(X[low_cardinality_nom_cols])
)
X_test_low_card_nom = pd.DataFrame(
    one_hot_encoder.transform(X_test[low_cardinality_nom_cols])
)

# Frequency encoding for nominal features with high cardinality
high_cardinality_nom_cols = [col for col in nominal_cols if X[col].nunique() >= 10]
for col in high_cardinality_nom_cols:
    freq_encoder = X[col].value_counts(normalize=True)
    X[col] = X[col].map(freq_encoder)
    X_test[col] = X_test[col].map(freq_encoder)

# Combine all features
X = pd.concat([X, X_low_card_nom], axis=1).drop(low_cardinality_nom_cols, axis=1)
X_test = pd.concat([X_test, X_test_low_card_nom], axis=1).drop(
    low_cardinality_nom_cols, axis=1
)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

# Define the model
model = LGBMClassifier()

# Train the model
model.fit(X_train, y_train)

# Predict on the validation set
valid_preds = model.predict_proba(X_valid)[:, 1]

# Evaluate the model
roc_auc = roc_auc_score(y_valid, valid_preds)
print(f"Validation ROC AUC Score: {roc_auc}")

# Predict on the test set
test_preds = model.predict_proba(X_test)[:, 1]

# Save the predictions to a CSV file
output = pd.DataFrame({"id": test_data.id, "target": test_preds})
output.to_csv("./working/submission.csv", index=False)
