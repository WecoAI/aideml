import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Load the data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Prepare the data
X_train, X_val, y_train, y_val = train_test_split(
    train_data["text"], train_data["target"], test_size=0.2, random_state=42
)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict on the validation set
val_predictions = model.predict(X_val_tfidf)

# Evaluate the model
f1 = f1_score(y_val, val_predictions)
print(f"F1 Score on the validation set: {f1}")

# Predict on the test set and save the submission
X_test_tfidf = vectorizer.transform(test_data["text"])
test_predictions = model.predict(X_test_tfidf)
submission = pd.DataFrame({"id": test_data["id"], "target": test_predictions})
submission.to_csv("./working/submission.csv", index=False)
