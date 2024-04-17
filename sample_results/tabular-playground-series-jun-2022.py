import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the data
data = pd.read_csv("./input/data.csv")

# Identify columns with missing values
cols_with_missing = [col for col in data.columns if data[col].isnull().any()]

# Initialize the SimpleImputer with mean strategy
imputer = SimpleImputer(strategy="mean")

# Fit the imputer on the dataset and transform the dataset
data[cols_with_missing] = imputer.fit_transform(data[cols_with_missing])

# Load the sample submission to find the missing values
sample_submission = pd.read_csv("./input/sample_submission.csv")

# Extract row and column to impute from the sample submission
sample_submission[["row_id", "column"]] = sample_submission["row-col"].str.split(
    "-", expand=True
)
sample_submission["row_id"] = sample_submission["row_id"].astype(int)

# Calculate the RMSE on the known missing values
original_values = []
imputed_values = []
for index, row in sample_submission.iterrows():
    original_values.append(row["value"])
    imputed_values.append(data.at[row["row_id"], row["column"]])

rmse = sqrt(mean_squared_error(original_values, imputed_values))
print(f"Validation RMSE: {rmse}")

# Prepare the submission file
sample_submission["value"] = imputed_values
sample_submission[["row-col", "value"]].to_csv("./working/submission.csv", index=False)
