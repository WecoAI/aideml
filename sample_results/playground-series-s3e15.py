import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import SimpleImputer
import numpy as np

# Load the data
data = pd.read_csv("./input/data.csv")

# Select only numeric columns for imputation
numeric_columns = data.select_dtypes(include=[np.number]).columns
X = data[numeric_columns].drop(columns=["x_e_out [-]"])
y = data["x_e_out [-]"]

# Split the data into training and NaN sets
train_data = data.dropna(subset=["x_e_out [-]"])
nan_data = data[data["x_e_out [-]"].isna()]

# Impute missing values in features with mean
imputer = SimpleImputer(strategy="mean")
X_train_imputed = imputer.fit_transform(
    train_data[numeric_columns].drop(columns=["x_e_out [-]"])
)
y_train = train_data["x_e_out [-]"]

# Train the linear regression model using cross-validation
model = LinearRegression()
kf = KFold(n_splits=10, shuffle=True, random_state=1)
rmse_scores = cross_val_score(
    model, X_train_imputed, y_train, scoring="neg_root_mean_squared_error", cv=kf
)
print(f"10-fold CV RMSE: {-np.mean(rmse_scores):.4f} (+/- {np.std(rmse_scores):.4f})")

# Fit the model on the entire training set
model.fit(X_train_imputed, y_train)

# Load the test data
test_data = pd.read_csv("./input/sample_submission.csv")

# Prepare the test features
X_test = nan_data[numeric_columns].drop(columns=["x_e_out [-]"])
X_test_imputed = imputer.transform(X_test)

# Predict the missing values for the test set
nan_data["x_e_out [-]"] = model.predict(X_test_imputed)

# Merge predictions back into the test data
test_data = test_data.merge(nan_data[["id", "x_e_out [-]"]], on="id", how="left")

# Save the submission file
test_data.to_csv("./working/submission.csv", index=False)
