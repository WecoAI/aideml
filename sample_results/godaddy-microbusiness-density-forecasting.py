import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the data
train_data = pd.read_csv("./input/train.csv")
census_data = pd.read_csv("./input/census_starter.csv")
test_data = pd.read_csv("./input/test.csv")

# Merge train and test data with census data
train_data = train_data.merge(census_data, on="cfips", how="left")
test_data = test_data.merge(census_data, on="cfips", how="left")

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy="median")

# Columns to be used as features
feature_columns = train_data.select_dtypes(exclude=["object", "datetime"]).columns.drop(
    "microbusiness_density"
)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, feature_columns),
    ]
)

# Define the model
model = RandomForestRegressor(random_state=0)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


# Define SMAPE function
def smape(actual, predicted):
    denominator = (abs(actual) + abs(predicted)) / 2.0
    diff = abs(predicted - actual) / denominator
    diff[denominator == 0] = 0.0
    return 100 * diff.mean()


smape_scorer = make_scorer(smape, greater_is_better=False)

# Define the grid of hyperparameters to search
param_grid = {
    "model__n_estimators": [50, 100, 150],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    my_pipeline, param_grid=param_grid, cv=3, scoring=smape_scorer, n_jobs=-1
)

# Fit the grid search to the data
grid_search.fit(train_data[feature_columns], train_data["microbusiness_density"])

# Print the best parameters and the corresponding SMAPE score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best SMAPE score: {-grid_search.best_score_}")

# Fit the model with the best parameters and make predictions on the test set
best_pipeline = grid_search.best_estimator_
best_pipeline.fit(train_data[feature_columns], train_data["microbusiness_density"])
test_preds = best_pipeline.predict(test_data[feature_columns])

# Save test predictions to file
output = pd.DataFrame({"row_id": test_data.row_id, "microbusiness_density": test_preds})
output.to_csv("./working/submission.csv", index=False)
