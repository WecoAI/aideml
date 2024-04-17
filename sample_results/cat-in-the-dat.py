import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Separate features and target
X = train_data.drop(["id", "target"], axis=1)
y = train_data["target"]
X_test = test_data.drop(["id"], axis=1)

# Identify categorical features
cat_features = [col for col in X.columns if X[col].dtype == "object"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the CatBoostClassifier with a smaller number of iterations for faster grid search
model = CatBoostClassifier(
    iterations=100,  # Reduced number of iterations for grid search
    learning_rate=0.1,
    depth=4,
    loss_function="Logloss",
    early_stopping_rounds=10,
    verbose=False,
)

# Fit the model on the training data
model.fit(X_train, y_train, cat_features=cat_features)

# Predict on the validation set
val_pred = model.predict_proba(X_val)[:, 1]

# Calculate the ROC AUC score
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {val_auc}")

# Train the final model on the full dataset with more iterations
final_model = CatBoostClassifier(
    iterations=1000,  # Increased number of iterations for final training
    learning_rate=0.1,
    depth=4,
    loss_function="Logloss",
    early_stopping_rounds=10,
    verbose=False,
)

final_model.fit(X, y, cat_features=cat_features)

# Predict on the test set
test_pred = final_model.predict_proba(X_test)[:, 1]

# Save the predictions to a CSV file
submission = pd.DataFrame({"id": test_data["id"], "target": test_pred})
submission.to_csv("./working/submission.csv", index=False)
