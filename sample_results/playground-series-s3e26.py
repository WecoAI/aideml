import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Separate features and target
X = train_data.drop(["Status", "id"], axis=1)
y = train_data["Status"]
X_test = test_data.drop("id", axis=1)

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
model = RandomForestClassifier(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Split data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

# Preprocessing of training data, fit model
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict_proba(X_valid)

# Evaluate the model
score = log_loss(pd.get_dummies(y_valid), preds)
print("Log Loss:", score)

# Preprocessing of test data, fit model
test_preds = clf.predict_proba(X_test)

# Generate submission file
output = pd.DataFrame(
    {
        "id": test_data.id,
        "Status_C": test_preds[:, 0],
        "Status_CL": test_preds[:, 1],
        "Status_D": test_preds[:, 2],
    }
)
output.to_csv("./working/submission.csv", index=False)
