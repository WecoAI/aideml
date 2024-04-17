import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import lightgbm as lgb

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Prepare the data
X = train_data.drop(["Id", "quality"], axis=1)
y = train_data["quality"]
X_test = test_data.drop("Id", axis=1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
val_predictions = model.predict(X_val)
val_predictions = np.round(val_predictions).astype(int)  # Round to the nearest integer

# Evaluate the model
kappa_score = cohen_kappa_score(y_val, val_predictions, weights="quadratic")
print(f"Quadratic Weighted Kappa score on validation set: {kappa_score}")

# Make predictions on the test set
test_predictions = model.predict(X_test)
test_predictions = np.round(test_predictions).astype(
    int
)  # Round to the nearest integer

# Prepare the submission file
submission = pd.DataFrame({"Id": test_data["Id"], "quality": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
