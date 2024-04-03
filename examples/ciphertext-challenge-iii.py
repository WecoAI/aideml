import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
sample_submission = pd.read_csv("./input/sample_submission.csv")

# Split the training data into training and validation sets
train_texts, val_texts, train_indices, val_indices = train_test_split(
    train_df["text"], train_df["index"], test_size=0.1, random_state=42
)

# Placeholder for the predictions
val_predictions = [0] * len(val_texts)

# TODO: Implement the decryption algorithm here
# For now, we are just using a placeholder prediction
# In a real scenario, this is where the decryption logic would be applied

# Evaluate the accuracy of the predictions
accuracy = accuracy_score(val_indices, val_predictions)
print(f"Validation accuracy: {accuracy}")

# Prepare the submission file
test_predictions = [0] * len(test_df)
submission = pd.DataFrame(
    {"ciphertext_id": test_df["ciphertext_id"], "index": test_predictions}
)
submission.to_csv("./working/submission.csv", index=False)
