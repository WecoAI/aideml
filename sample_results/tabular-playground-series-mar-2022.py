import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Preprocess the data
train_data["time"] = pd.to_datetime(train_data["time"])
train_data["hour"] = train_data["time"].dt.hour
train_data["weekday"] = train_data["time"].dt.weekday
train_data["month"] = train_data["time"].dt.month

test_data["time"] = pd.to_datetime(test_data["time"])
test_data["hour"] = test_data["time"].dt.hour
test_data["weekday"] = test_data["time"].dt.weekday
test_data["month"] = test_data["time"].dt.month

# Prepare features and target
X = train_data[["x", "y", "direction", "hour", "weekday", "month"]]
X = pd.get_dummies(X, columns=["direction"], drop_first=True)
y = train_data["congestion"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
print(f"Mean Absolute Error: {mae}")

# Prepare test data and make predictions
X_test = test_data[["x", "y", "direction", "hour", "weekday", "month"]]
X_test = pd.get_dummies(X_test, columns=["direction"], drop_first=True)
test_predictions = model.predict(X_test)

# Save the predictions to a CSV file
submission = pd.DataFrame(
    {"row_id": test_data["row_id"], "congestion": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
