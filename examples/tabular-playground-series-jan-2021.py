import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Split the data into features and target
X = train_data.drop(["id", "target"], axis=1)
y = train_data["target"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Gradient Boosting Regressor
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict on the validation set and calculate RMSE
y_pred_val = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred_val, squared=False)
print(f"Validation RMSE: {rmse}")

# Predict on the test set
test_features = test_data.drop("id", axis=1)
test_predictions = model.predict(test_features)

# Save the predictions to a CSV file
submission = pd.DataFrame({"id": test_data["id"], "target": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
