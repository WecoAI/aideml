import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load the data
train_data = pd.read_csv("./input/train.csv")

# Prepare the features and labels
X = train_data["comment_text"]
y = train_data.iloc[:, 2:]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)

# Train a logistic regression model for each label
scores = []
for label in y.columns:
    lr = LogisticRegression(C=1.0, solver="liblinear")
    lr.fit(X_train_tfidf, y_train[label])
    y_pred = lr.predict_proba(X_val_tfidf)[:, 1]
    score = roc_auc_score(y_val[label], y_pred)
    scores.append(score)
    print(f"ROC AUC for {label}: {score}")

# Calculate the mean column-wise ROC AUC
mean_auc = sum(scores) / len(scores)
print(f"Mean column-wise ROC AUC: {mean_auc}")
