import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Preprocess the data
features = train_data.columns.drop(["id", "failure"])
X = train_data[features]
y = train_data["failure"]
X_test = test_data[features]

# Fill missing values with median for numerical columns
num_cols = X.select_dtypes(exclude="object").columns
imputer = SimpleImputer(strategy="median")
X[num_cols] = imputer.fit_transform(X[num_cols])
X_test[num_cols] = imputer.transform(X_test[num_cols])

# One-hot encode categorical features
cat_cols = X.select_dtypes(include="object").columns
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
X_encoded = pd.DataFrame(
    encoder.fit_transform(X[cat_cols]), columns=encoder.get_feature_names_out(cat_cols)
)
X_test_encoded = pd.DataFrame(
    encoder.transform(X_test[cat_cols]), columns=encoder.get_feature_names_out(cat_cols)
)

# One-hot encoding removed index; put it back
X_encoded.index = X.index
X_test_encoded.index = X_test.index

# Remove categorical columns (will replace with one-hot encoding)
num_X = X.drop(cat_cols, axis=1)
num_X_test = X_test.drop(cat_cols, axis=1)

# Add one-hot encoded columns to numerical features
X_preprocessed = pd.concat([num_X, X_encoded], axis=1)
X_test_preprocessed = pd.concat([num_X_test, X_test_encoded], axis=1)

# Convert all feature names to strings to avoid TypeError
X_preprocessed.columns = X_preprocessed.columns.astype(str)
X_test_preprocessed.columns = X_test_preprocessed.columns.astype(str)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=0
)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
val_predictions = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_predictions)
print(f"Validation ROC AUC Score: {val_auc}")

# Predict on test data
test_predictions = model.predict_proba(X_test_preprocessed)[:, 1]

# Save the predictions to a CSV file
output = pd.DataFrame({"id": test_data.id, "failure": test_predictions})
output.to_csv("./working/submission.csv", index=False)
