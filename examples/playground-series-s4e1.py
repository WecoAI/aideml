import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Separate features and target
X = train_data.drop(["Exited", "id", "CustomerId", "Surname"], axis=1)
y = train_data["Exited"]
X_test = test_data.drop(["id", "CustomerId", "Surname"], axis=1)

# Preprocessing for numerical data
numerical_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, X.select_dtypes(exclude=["object"]).columns),
        ("cat", categorical_transformer, X.select_dtypes(include=["object"]).columns),
    ]
)

# Define the model
model = GradientBoostingClassifier()

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Split data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Preprocessing of training data, fit model
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict_proba(X_valid)[:, 1]

# Evaluate the model
score = roc_auc_score(y_valid, preds)
print(f"ROC AUC score: {score}")

# Preprocessing of test data, fit model
preds_test = clf.predict_proba(X_test)[:, 1]

# Save test predictions to file
output = pd.DataFrame({"id": test_data.id, "Exited": preds_test})
output.to_csv("./working/submission.csv", index=False)
