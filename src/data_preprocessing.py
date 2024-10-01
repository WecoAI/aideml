import pandas as pd

def preprocess_data(data):
    # Check if the target column exists in the DataFrame
    if 'SalePrice' in data.columns:
        # Drop the target variable from features for training data
        X = data.drop(columns=['SalePrice'])
        y = data['SalePrice']
        return X, y
    else:
        # For test data, just return the features without any target variable
        X = data
        return X, None  # Return None for y as it's not available for test data
