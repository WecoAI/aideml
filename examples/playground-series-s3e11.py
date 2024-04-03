import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import numpy as np

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Prepare the data
X = train_data.drop(["id", "cost"], axis=1)
y = train_data["cost"]
X_test = test_data.drop("id", axis=1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_val)
y_pred_test = model.predict(X_test)

# Calculate the RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
print(f"Validation RMSLE: {rmsle}")

# Prepare the submission file
submission = pd.DataFrame({"id": test_data["id"], "cost": y_pred_test})
submission.to_csv("./working/submission.csv", index=False)
