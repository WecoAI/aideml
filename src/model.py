from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

def train_model(data):
    """Train a model using the provided data."""
    print("Training model with the provided data...")
    
    # Assuming 'target' is the column you want to predict
    X = data.drop('target', axis=1)  # Features
    y = data['target']  # Target variable
    
    # Splitting the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = LinearRegression()  # You can choose other models based on your needs
    model.fit(X_train, y_train)
    
    print("Model trained successfully!")
    return model

def predict(model, new_data):
    """Make predictions using the trained model."""
    print("Making predictions with the model...")
    
    # Ensure new_data is a DataFrame
    if isinstance(new_data, pd.DataFrame):
        predictions = model.predict(new_data)  # Use the model to make predictions
    else:
        raise ValueError("new_data must be a Pandas DataFrame")

    return predictions
