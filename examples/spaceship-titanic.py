import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Separate target from predictors
y = train_data["Transported"]
X = train_data.drop(["Transported"], axis=1)

# Select categorical columns with relatively low cardinality
categorical_cols = [
    cname
    for cname in X.columns
    if X[cname].nunique() < 10 and X[cname].dtype == "object"
]

# Select numerical columns
numerical_cols = [
    cname for cname in X.columns if X[cname].dtype in ["int64", "float64"]
]

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy="median")

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
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
preds = clf.predict(X_valid)

# Evaluate the model
score = accuracy_score(y_valid, preds)
print("Accuracy:", score)

# Preprocessing of test data, fit model
preprocessed_test_data = clf.named_steps["preprocessor"].transform(test_data)

# Get test predictions
test_preds = clf.named_steps["model"].predict(preprocessed_test_data)

# Save test predictions to file
output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Transported": test_preds})
output.to_csv("./working/submission.csv", index=False)
