import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Separate features and target
X = train_data.drop(["Age", "id"], axis=1)
y = train_data["Age"]
test_X = test_data.drop(["id"], axis=1)


# Generate polynomial features for selected columns and ensure unique feature names by adding a prefix
def generate_poly_features(
    df, feature_names, degree=2, include_bias=False, prefix="poly_"
):
    poly_features = PolynomialFeatures(degree=degree, include_bias=include_bias)
    selected_features = df[feature_names]
    poly_features_array = poly_features.fit_transform(selected_features)
    poly_feature_names = [
        prefix + name for name in poly_features.get_feature_names_out(feature_names)
    ]
    return pd.DataFrame(poly_features_array, columns=poly_feature_names)


# Apply polynomial feature generation to both train and test datasets
poly_features_train = generate_poly_features(X, ["Length", "Diameter", "Height"])
poly_features_test = generate_poly_features(test_X, ["Length", "Diameter", "Height"])

# Concatenate the polynomial features with the original dataset
X_poly = pd.concat([X.reset_index(drop=True), poly_features_train], axis=1)
test_X_poly = pd.concat([test_X.reset_index(drop=True), poly_features_test], axis=1)

# Specify categorical features
cat_features = ["Sex"]

# Initialize KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize an empty list to store MAE for each fold
mae_scores = []

# Define hyperparameters
hyperparams = {
    "iterations": 1500,
    "learning_rate": 0.05,
    "depth": 8,
    "loss_function": "MAE",
    "cat_features": cat_features,
    "verbose": 0,
}

# Loop over each fold
for train_index, test_index in kf.split(X_poly):
    X_train, X_val = X_poly.iloc[train_index], X_poly.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    # Initialize CatBoostRegressor with hyperparameters
    model = CatBoostRegressor(**hyperparams)

    # Train the model
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=0,
    )

    # Predict on validation set
    predictions = model.predict(X_val)

    # Calculate and print MAE
    mae = mean_absolute_error(y_val, predictions)
    mae_scores.append(mae)

# Print the average MAE across all folds
print(f"Average MAE across all folds: {sum(mae_scores) / len(mae_scores)}")

# Predict on the test set
test_predictions = model.predict(test_X_poly)

# Prepare submission file
submission_df = pd.DataFrame({"id": test_data["id"], "Age": test_predictions})
submission_df.to_csv("./working/submission.csv", index=False)
