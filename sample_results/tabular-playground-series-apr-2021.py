import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Features and target
X = train_data.drop(["Survived", "PassengerId", "Name", "Ticket", "Cabin"], axis=1)
y = train_data["Survived"]

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy="median")

# Preprocessing for categorical data
categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, ["Age", "Fare"]),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Cross-validation scores
scores = cross_val_score(clf, X, y, cv=10, scoring="accuracy")
print(f"Average cross-validation score: {scores.mean():.4f}")

# Preprocessing of test data, fit model
clf.fit(X, y)
test_X = test_data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
test_preds = clf.predict(test_X)

# Save test predictions to file
output = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": test_preds})
output.to_csv("./working/submission.csv", index=False)
