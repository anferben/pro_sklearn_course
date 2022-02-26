import pandas as pd

from sklearn.linear_model import RANSACRegressor, HuberRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def data_split():
    # Loading the data
    dataset = pd.read_csv('./data/felicidad_corrupt.csv')

    # Separating label from reatures and splitting the data
    X = dataset.drop(['country', 'score'], axis=1)  
    y = dataset['score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def train(X_train, X_test, y_train, y_test):
    # Defining the estimators to use
    estimators = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)
    }

    # Training each estimator and printing results
    for name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        error = mean_squared_error(y_test, y_pred)

        print('=' * 64)
        print(name)
        print(f'MSE: {error}')


def main():
    X_train, X_test, y_train, y_test = data_split()
    train(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()