import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Separate features and target
X = train_data.drop(["id", "loss"], axis=1)
y = train_data["loss"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Initialize the model
model = GradientBoostingRegressor(random_state=42)

# Fit the model
model.fit(X_train_scaled, y_train)

# Predict on the validation set
y_pred = model.predict(X_val_scaled)

# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse}")

# Prepare the test set
X_test = test_data.drop("id", axis=1)
X_test_scaled = scaler.transform(X_test)

# Predict on the test set
test_predictions = model.predict(X_test_scaled)

# Create the submission file
submission = pd.DataFrame({"id": test_data["id"], "loss": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
