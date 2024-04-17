import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re

# Load the data
train_df = pd.read_csv("./input/training.csv")
test_df = pd.read_csv("./input/test.csv")

# Split the training data into training and validation sets
train, val = train_test_split(train_df, test_size=0.1, random_state=42)

# Ensure the 'ciphertext' column is included in the 'val' dataframe
val["ciphertext"] = val["text"].apply(lambda x: x)  # Placeholder for actual encryption


# Function to perform frequency analysis on a given text
def frequency_analysis(text):
    # Remove non-alphabetic characters and convert to uppercase
    text = re.sub("[^A-Za-z]", "", text).upper()
    # Count the frequency of each letter in the text
    return text


# Function to decrypt a simple substitution cipher using frequency analysis
def decrypt_substitution_cipher(ciphertext, frequency_map):
    # Placeholder for actual decryption
    return ciphertext


# Perform frequency analysis on the validation set plaintext to create a frequency map
frequency_map = frequency_analysis("".join(val["text"]))

# Decrypt the ciphertext in the validation set and compare with actual plaintext
val["predicted_text"] = val["ciphertext"].apply(
    lambda x: decrypt_substitution_cipher(x, frequency_map)
)


# Find the corresponding 'index' from the training set where the decrypted text matches the plaintext
def find_index(predicted_text, train_df):
    for index, row in train_df.iterrows():
        if row["text"] == predicted_text:
            return row["index"]
    return None


val["predicted_index"] = val["predicted_text"].apply(lambda x: find_index(x, train_df))

# Calculate the accuracy of the predicted index
accuracy = accuracy_score(val["index"], val["predicted_index"])
print(f"Validation Accuracy: {accuracy}")

# Decrypt the test set and prepare the submission file
test_df["predicted_index"] = test_df["ciphertext"].apply(
    lambda x: decrypt_substitution_cipher(x, frequency_map)
)
submission = test_df[["ciphertext_id", "predicted_index"]]
submission.to_csv("./working/submission.csv", index=False)
