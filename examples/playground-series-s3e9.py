import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Prepare the data
X = train_data.drop(["id", "Strength"], axis=1)
y = train_data["Strength"]
X_test = test_data.drop("id", axis=1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict on validation set
y_pred_val = model.predict(X_val)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"Validation RMSE: {rmse}")

# Predict on test set
test_predictions = model.predict(X_test)

# Save the predictions to a CSV file
submission = pd.DataFrame({"id": test_data["id"], "Strength": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
