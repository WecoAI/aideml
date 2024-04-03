import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Separate features and target
X = train_data.drop(["Attrition", "id"], axis=1)
y = train_data["Attrition"]
X_test = test_data.drop("id", axis=1)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

# Create the preprocessing pipelines for both numeric and categorical data
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Create a pipeline that combines the preprocessor with a classifier
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(solver="liblinear")),
    ]
)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict probabilities on the validation set
y_pred_proba = model.predict_proba(X_val)[:, 1]

# Calculate the AUC
auc = roc_auc_score(y_val, y_pred_proba)
print(f"Validation AUC: {auc}")

# Predict probabilities on the test set
test_pred_proba = model.predict_proba(X_test)[:, 1]

# Create a submission file
submission = pd.DataFrame(
    {"EmployeeNumber": test_data["id"], "Attrition": test_pred_proba}
)
submission.to_csv("./working/submission.csv", index=False)
