import pandas as pd

def preprocess_data(data):
    """Preprocess the input DataFrame."""
    # Ensure you're only applying string methods to string columns
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = data[column].str.lower().str.strip()
    
    # Check if 'SalePrice' exists
    if 'SalePrice' not in data.columns:
        raise ValueError("The expected target column 'SalePrice' is not found in the DataFrame.")
    
    # Separate features and target variable
    X = data.drop('SalePrice', axis=1)  # Use 'SalePrice' as the target column
    y = data['SalePrice']  # Use 'SalePrice' as the target column
    
    return X, y
