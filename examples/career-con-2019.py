import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the data
X_train = pd.read_csv("./input/X_train.csv")
y_train = pd.read_csv("./input/y_train.csv")
X_test = pd.read_csv("./input/X_test.csv")

# Merge the sensor data with the target variable
train_data = X_train.merge(y_train, on="series_id", how="inner")

# Drop non-feature columns
features = train_data.drop(
    ["row_id", "series_id", "measurement_number", "group_id", "surface"], axis=1
)
labels = train_data["surface"]

# Normalize the feature data
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Initialize the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict on the validation set
y_pred = rf.predict(X_val)

# Calculate the accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")

# Prepare the test data
test_features = X_test.drop(["row_id", "series_id", "measurement_number"], axis=1)
test_features = scaler.transform(test_features)

# Predict on the test set
test_predictions = rf.predict(test_features)

# Save the predictions to a CSV file
submission = pd.DataFrame(
    {"series_id": X_test["series_id"], "surface": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
