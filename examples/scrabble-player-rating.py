import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the data
games = pd.read_csv("./input/games.csv")
turns = pd.read_csv("./input/turns.csv")
train = pd.read_csv("./input/train.csv")

# Merge the datasets on game_id
merged_data = pd.merge(train, games, on="game_id")
merged_data = pd.merge(
    merged_data,
    turns.groupby("game_id").agg({"points": "sum"}).reset_index(),
    on="game_id",
)

# Prepare the features and target variable
X = merged_data[["game_duration_seconds", "winner", "points"]]
y = merged_data["rating"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_val)

# Calculate the RMSE
rmse = sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse}")

# Prepare the test set
test = pd.read_csv("./input/test.csv")
test_merged = pd.merge(test, games, on="game_id")
test_merged = pd.merge(
    test_merged,
    turns.groupby("game_id").agg({"points": "sum"}).reset_index(),
    on="game_id",
)
X_test = test_merged[["game_duration_seconds", "winner", "points"]]

# Predict on the test set
test["rating"] = model.predict(X_test)

# Save the predictions to a CSV file
test[["game_id", "rating"]].to_csv("./working/submission.csv", index=False)
