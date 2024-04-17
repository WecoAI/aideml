import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import LabelEncoder


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)


# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Convert date to datetime and extract features
for df in [train, test]:
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek  # Extract day of the week
    # Create interaction terms
    df["country_store"] = df["country"] + "_" + df["store"]
    df["country_product"] = df["country"] + "_" + df["product"]
    df["store_product"] = df["store"] + "_" + df["product"]

# Encode new categorical features
label_encoders = {}
for col in ["country_store", "country_product", "store_product"]:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

# Define categorical features including new interaction terms
cat_features = [
    "country",
    "store",
    "product",
    "country_store",
    "country_product",
    "store_product",
]

# Prepare data for training
X = train.drop(["num_sold", "date", "id"], axis=1)
y = train["num_sold"]

# Cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
smape_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_valid = X.iloc[train_index], X.iloc[test_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

    model = CatBoostRegressor(
        iterations=2000,  # Increase iterations
        learning_rate=0.05,  # Decrease learning rate
        depth=8,  # Increase depth
        loss_function="MAE",
        cat_features=cat_features,
        verbose=200,
    )
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

    preds = model.predict(X_valid)
    score = smape(y_valid, preds)
    smape_scores.append(score)

print(f"Average SMAPE: {np.mean(smape_scores)}")

# Prepare test data and make predictions
test_data = test.drop(["date", "id"], axis=1)
predictions = model.predict(test_data)

# Save submission
submission = pd.DataFrame({"id": test["id"], "num_sold": predictions})
submission.to_csv("./working/submission.csv", index=False)
