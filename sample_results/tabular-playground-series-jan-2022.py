import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
from sklearn.metrics import make_scorer

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")


# Preprocess the data
def preprocess_data(data):
    data["date"] = pd.to_datetime(data["date"])
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["dayofweek"] = data["date"].dt.dayofweek
    data["is_weekend"] = data["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
    data["day_of_year"] = data["date"].dt.dayofyear
    data["sin_day_of_year"] = np.sin(2 * np.pi * data["day_of_year"] / 365.25)
    data["cos_day_of_year"] = np.cos(2 * np.pi * data["day_of_year"] / 365.25)
    data["country_month"] = data["country"] + "_" + data["month"].astype(str)
    data = pd.get_dummies(
        data, columns=["country", "store", "product", "country_month"]
    )
    return data


train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Prepare the data for LightGBM
X = train_data.drop(["num_sold", "date", "row_id"], axis=1)
y = train_data["num_sold"]


# Define SMAPE function
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)


# Custom scorer for cross-validation
smape_scorer = make_scorer(smape, greater_is_better=False)

# Define the parameter grid
param_grid = {
    "num_leaves": [31, 50, 70],
    "learning_rate": [0.1, 0.05, 0.01],
    "n_estimators": [100, 200, 500],
}

# Perform grid search with 10-fold cross-validation
gbm = lgb.LGBMRegressor(objective="regression", metric="mae", verbose=-1)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=gbm, param_grid=param_grid, cv=kf, scoring=smape_scorer, verbose=1
)
grid_search.fit(X, y)

# Train the model on full data with the best parameters
best_params = grid_search.best_params_
gbm_best = lgb.LGBMRegressor(**best_params)
gbm_best.fit(X, y)

# Predict on test set and save the submission file
X_test = test_data.drop(["date", "row_id"], axis=1)
test_data["num_sold"] = gbm_best.predict(X_test)
submission = test_data[["row_id", "num_sold"]]
submission.to_csv("./working/submission.csv", index=False)

# Print the evaluation metric
print("Best SMAPE (GridSearchCV):", -grid_search.best_score_)
