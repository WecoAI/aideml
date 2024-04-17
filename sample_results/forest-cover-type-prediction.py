import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Prepare the data
X = train_data.drop(["Id", "Cover_Type"], axis=1)
y = train_data["Cover_Type"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Validate the model
y_pred = rf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")

# Predict on test data
test_ids = test_data["Id"]
test_data = test_data.drop("Id", axis=1)
test_predictions = rf.predict(test_data)

# Save the predictions
submission = pd.DataFrame({"Id": test_ids, "Cover_Type": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
