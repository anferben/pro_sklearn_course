import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


def data_split(data):
    X = data.drop(['country', 'rank', 'score'], axis=1)
    y = data['score']

    return X, y


def model_train(X, y):
    reg = RandomForestRegressor()

    parameters = {
        'n_estimators': range(4, 16),
        'criterion': ['mse', 'mae'],
        'max_depth': range(2, 11)
    }

    rand_est = RandomizedSearchCV(reg, param_distributions=parameters, n_iter=10,
                                  cv=3, scoring='neg_mean_absolute_error')

    model = rand_est.fit(X, y)

    #print(rand_est.best_estimator_)
    print(rand_est.best_params_)
    print(rand_est.predict(X.loc[[0]]))


def main():
    # Loading the data
    data = pd.read_csv('./data/felicidad.csv')

    X, y = data_split(data)
    model_train(X, y)


if __name__ == '__main__':
    main()