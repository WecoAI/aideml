import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Separate target from predictors
y = train_data.target
X = train_data.drop(["target", "id"], axis=1)
X_test = test_data.drop(["id"], axis=1)

# One-hot encode the categorical data
cat_cols = [col for col in X.columns if "cat" in col]
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X[cat_cols]))
X_test_encoded = pd.DataFrame(encoder.transform(X_test[cat_cols]))

# One-hot encoding removed index; put it back
X_encoded.index = X.index
X_test_encoded.index = X_test.index

# Remove categorical columns (will replace with one-hot encoding)
num_X = X.drop(cat_cols, axis=1)
num_X_test = X_test.drop(cat_cols, axis=1)

# Add one-hot encoded columns to numerical features
X_final = pd.concat([num_X, X_encoded], axis=1)
X_test_final = pd.concat([num_X_test, X_test_encoded], axis=1)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X_final, y, train_size=0.8, test_size=0.2, random_state=0
)

# Define the model
model = LGBMClassifier()

# Train the model
model.fit(X_train, y_train)

# Predict on the validation set
val_predictions = model.predict_proba(X_valid)[:, 1]

# Calculate the AUC score
val_auc = roc_auc_score(y_valid, val_predictions)

# Print the AUC score
print(f"Validation AUC: {val_auc}")

# Predict on the test set
test_predictions = model.predict_proba(X_test_final)[:, 1]

# Save the predictions to a CSV file
output = pd.DataFrame({"id": test_data.id, "target": test_predictions})
output.to_csv("./working/submission.csv", index=False)
