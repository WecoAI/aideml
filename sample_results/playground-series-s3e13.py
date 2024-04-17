import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Prepare the data
X = train_data.drop(["id", "prognosis"], axis=1)
y = train_data["prognosis"]
X_test = test_data.drop(["id"], axis=1)
test_ids = test_data["id"]

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train the model
model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred_proba = model.predict_proba(X_val)

# Select the top 3 predictions for each sample
top3_preds = pd.DataFrame(y_val_pred_proba).apply(
    lambda x: label_encoder.inverse_transform(x.argsort()[-3:][::-1]), axis=1
)


# Evaluate the model using MPA@3
def mpa_at_k(y_true, y_pred, k=3):
    score = 0.0
    for true, pred in zip(y_true, y_pred):
        try:
            index = list(pred).index(true)
            score += 1.0 / (index + 1)
        except ValueError:
            continue
    return score / len(y_true)


# Calculate the MPA@3 score
y_val_true = label_encoder.inverse_transform(y_val)
mpa_score = mpa_at_k(y_val_true, top3_preds)
print(f"MPA@3 score on the validation set: {mpa_score}")

# Predict on the test set
y_test_pred_proba = model.predict_proba(X_test)
top3_test_preds = pd.DataFrame(y_test_pred_proba).apply(
    lambda x: label_encoder.inverse_transform(x.argsort()[-3:][::-1]), axis=1
)

# Prepare the submission file
submission = pd.DataFrame(
    {
        "id": test_ids,
        "prognosis": [" ".join(map(str, preds)) for preds in top3_test_preds],
    }
)
submission.to_csv("./working/submission.csv", index=False)
