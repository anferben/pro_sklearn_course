import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score


def model_train(X, y):
    model = DecisionTreeRegressor()
    score = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
    print(np.abs(np.mean(score)))


def main():
    # Loading the data
    dataset = pd.read_csv('./data/felicidad.csv')

    X = dataset.drop(['country', 'score'], axis=1)
    y = dataset['score']

    model_train(X, y)


if __name__ == '__main__':
    main()