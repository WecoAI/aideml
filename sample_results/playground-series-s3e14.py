import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Prepare the data
X = train_data.drop(["id", "yield"], axis=1)
y = train_data["yield"]
X_test = test_data.drop("id", axis=1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Initialize base models
estimators = [
    (
        "gbr",
        GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
        ),
    ),
    ("rf", RandomForestRegressor(n_estimators=200, random_state=42)),
    ("lr", LinearRegression()),
]

# Initialize the StackingRegressor with a RidgeCV final estimator
stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())

# Train the StackingRegressor on the scaled training data
stacking_regressor.fit(X_train_scaled, y_train)

# Predict on the scaled validation set using the StackingRegressor
y_val_pred = stacking_regressor.predict(X_val_scaled)

# Evaluate the model
mae = mean_absolute_error(y_val, y_val_pred)
print(f"Mean Absolute Error on validation set with StackingRegressor: {mae}")

# Train the StackingRegressor on the full scaled training data and predict on the scaled test set
stacking_regressor.fit(scaler.transform(X), y)
test_predictions = stacking_regressor.predict(X_test_scaled)

# Save the predictions to a CSV file
submission = pd.DataFrame({"id": test_data["id"], "yield": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
