import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the training data
train_data = pd.read_csv("./input/train.tsv", sep="\t")

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_data["Phrase"], train_data["Sentiment"], test_size=0.2, random_state=42
)

# Initialize a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train.astype(str))

# Transform the validation data
X_val_tfidf = tfidf_vectorizer.transform(X_val.astype(str))

# Initialize the Logistic Regression model
logistic_regression_model = LogisticRegression(random_state=42)

# Train the model
logistic_regression_model.fit(X_train_tfidf, y_train)

# Predict the sentiments on the validation set
y_val_pred = logistic_regression_model.predict(X_val_tfidf)

# Calculate the accuracy on the validation set
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy}")

# Load the test data
test_data = pd.read_csv("./input/test.tsv", sep="\t")

# Preprocess the test data by filling NaN values with an empty string
test_data["Phrase"] = test_data["Phrase"].fillna("")

# Transform the test data using the same vectorizer
X_test_tfidf = tfidf_vectorizer.transform(test_data["Phrase"].astype(str))

# Predict the sentiments on the test set
test_predictions = logistic_regression_model.predict(X_test_tfidf)

# Prepare the submission file
submission = pd.DataFrame(
    {"PhraseId": test_data["PhraseId"], "Sentiment": test_predictions}
)

# Save the submission file
submission.to_csv("./working/submission.csv", index=False)
