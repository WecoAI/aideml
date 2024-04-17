import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
encoded_features = encoder.fit_transform(train_data[["Product ID", "Type"]])
encoded_test_features = encoder.transform(test_data[["Product ID", "Type"]])

# Add encoded features back to the dataframe
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())
train_data = train_data.join(encoded_df).drop(["Product ID", "Type"], axis=1)

encoded_test_df = pd.DataFrame(
    encoded_test_features, columns=encoder.get_feature_names_out()
)
test_data = test_data.join(encoded_test_df).drop(["Product ID", "Type"], axis=1)

# Split the data into features and target
X = train_data.drop(["Machine failure", "id"], axis=1)
y = train_data["Machine failure"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=3, scoring="roc_auc", n_jobs=-1
)

# Perform grid search
grid_search.fit(X_train, y_train)

# Get the best estimator
best_rf = grid_search.best_estimator_

# Predict on the validation set using the best estimator
y_pred_proba = best_rf.predict_proba(X_val)[:, 1]

# Calculate the AUC-ROC score
auc_roc = roc_auc_score(y_val, y_pred_proba)
print(f"AUC-ROC score: {auc_roc}")

# Predict on the test set using the best estimator
test_predictions = best_rf.predict_proba(test_data.drop("id", axis=1))[:, 1]

# Create the submission file
submission = pd.DataFrame({"id": test_data["id"], "Machine failure": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
