import pandas as pd
from data_processing import preprocess_data  # Adjust if you have this in a separate file

# Load your data
data = pd.read_csv('path/to/your/data.csv')  # Update with the correct path

# Check if 'SalePrice' exists
if 'SalePrice' not in data.columns:
    raise ValueError("The expected target column 'SalePrice' is not found in the DataFrame.")

# Preprocess the data
X, y = preprocess_data(data)

# Continue with your model training or other operations
