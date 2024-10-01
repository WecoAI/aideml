import pandas as pd
from src.data_preprocessing import preprocess_data  # Ensure the path is correct

# Load your data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Print column names for inspection
print("Train Data Columns:", train_data.columns.tolist())
print("Test Data Columns:", test_data.columns.tolist())

# Check if 'SalePrice' exists in the training data
if 'SalePrice' not in train_data.columns:
    raise ValueError("The expected target column 'SalePrice' is not found in the training DataFrame.")

# Preprocess the training data
X_train, y_train = preprocess_data(train_data)

# Optionally preprocess test data if necessary
X_test, y_test = preprocess_data(test_data)  # Ensure preprocess_data can handle test data appropriately

# Continue with your model training or other operations
