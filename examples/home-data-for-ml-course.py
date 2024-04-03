import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Separate target from predictors
y = train_data.SalePrice
X = train_data.drop(["SalePrice"], axis=1)

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
        ("num", numerical_transformer, X.select_dtypes(exclude=["object"]).columns),
        ("cat", categorical_transformer, X.select_dtypes(include=["object"]).columns),
    ]
)

# Define the model
model = GradientBoostingRegressor()

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Split data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, np.log(y_train))

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_squared_error(np.log(y_valid), preds, squared=False)
print("RMSE:", score)

# Preprocessing of test data, fit model
test_preds = my_pipeline.predict(test_data)

# Save test predictions to file
output = pd.DataFrame({"Id": test_data.Id, "SalePrice": np.exp(test_preds)})
output.to_csv("./working/submission.csv", index=False)
