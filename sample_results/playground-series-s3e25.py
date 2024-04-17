import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import median_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Load the dataset
train_data = pd.read_csv("./input/train.csv")

# Separate features and target
X = train_data.drop(["id", "Hardness"], axis=1)
y = train_data["Hardness"]

# Splitting the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create interaction terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_val_scaled = scaler.transform(X_val_poly)

# Initialize the SVR model
svr = SVR(kernel="rbf")

# Define the expanded parameter grid
param_grid = {
    "C": [0.1, 0.5, 1, 1.5, 2, 2.5, 3],
    "gamma": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    "epsilon": [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    svr, param_grid, cv=5, scoring="neg_median_absolute_error", verbose=1, n_jobs=-1
)

# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Best parameters
print(f"Best parameters: {grid_search.best_params_}")

# Predict on the validation set using the best estimator
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_val_scaled)

# Evaluate the model
medae = median_absolute_error(y_val, predictions)
print(f"Median Absolute Error: {medae}")

# Prepare submission
test_data = pd.read_csv("./input/test.csv")
X_test = test_data.drop(["id"], axis=1)
X_test_poly = poly.transform(X_test)
X_test_scaled = scaler.transform(X_test_poly)
test_predictions = best_model.predict(X_test_scaled)
submission = pd.DataFrame({"id": test_data["id"], "Hardness": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
