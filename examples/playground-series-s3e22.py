import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Identify common categorical columns
categorical_features = train_data.select_dtypes(include=["object"]).columns
common_categorical_features = [
    feature for feature in categorical_features if feature in test_data.columns
]

# Convert common categorical columns to 'category' data type
for feature in common_categorical_features:
    train_data[feature] = train_data[feature].astype("category")
    test_data[feature] = test_data[feature].astype("category")

# Separate features and target
X = train_data.drop(["id", "outcome"], axis=1)
y = train_data["outcome"]
X_test = test_data.drop(["id"], axis=1)

# Prepare for cross-validation
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
predictions = pd.DataFrame()
scores = []

# Perform 10-fold cross-validation
for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # Train the model
    model = lgb.LGBMClassifier(objective="multiclass", random_state=42)
    model.fit(
        X_train,
        y_train,
        categorical_feature=common_categorical_features,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=50,
        verbose=False,
    )

    # Make predictions
    y_pred = model.predict(X_valid)

    # Calculate the F1 score
    score = f1_score(y_valid, y_pred, average="micro")
    scores.append(score)

    # Predict on test set
    predictions[f"fold_{fold_n}"] = model.predict(X_test)

# Print the average F1 score across all folds
print(f"Average F1-Score: {sum(scores) / len(scores)}")

# Prepare submission file
submission = pd.DataFrame()
submission["id"] = test_data["id"]
submission["outcome"] = predictions.mode(axis=1)[0]
submission.to_csv("./working/submission.csv", index=False)
