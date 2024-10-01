import pandas as pd
from src.data_preprocessing import preprocess_data
from src.model import train_model, predict
from src.evaluation import evaluate_model

def main():
    # Load data
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')

    # Preprocess data
    X_train, y_train = preprocess_data(train_data)
    X_test, y_test = preprocess_data(test_data)

    # Train model
    model = train_model(X_train, y_train)

    # Make predictions
    predictions = predict(model, X_test)

    # Evaluate model
    mae = evaluate_model(y_test, predictions)
    print(f'Mean Absolute Error: {mae}')

if __name__ == '__main__':
    main()
