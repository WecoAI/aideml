import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Separate features and target
X = train_data.drop(columns=["Id", "Cover_Type"])
y = train_data["Cover_Type"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the validation set and calculate accuracy
val_predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {accuracy}")

# Predict on the test set
test_predictions = model.predict(test_data.drop(columns=["Id"]))

# Save the predictions to a CSV file
submission = pd.DataFrame({"Id": test_data["Id"], "Cover_Type": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
