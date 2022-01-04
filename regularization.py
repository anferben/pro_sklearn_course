import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset.describe())

    X = dataset[['gdp', 'family', 'lifexp', 'freedom',
                'corruption', 'generosity', 'dystopia']]
    y = dataset['score']

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    linear_model = LinearRegression().fit(X_train, y_train)
    y_linear_pred = linear_model.predict(X_test)

    lasso_model = Lasso(alpha=0.02).fit(X_train, y_train)
    y_lasso_pred = lasso_model.predict(X_test)

    ridge_model = Ridge(alpha=1).fit(X_train, y_train)
    y_ridge_pred = ridge_model.predict(X_test)
    
    linear_loss = mean_squared_error(y_test, y_linear_pred)
    print('LINEAR LOSS: ', linear_loss)

    linear_lasso = mean_squared_error(y_test, y_lasso_pred)
    print('LASSO LOSS: ', linear_lasso)

    ridge_loss = mean_squared_error(y_test, y_ridge_pred)
    print('RIDGE LOSS: ', ridge_loss)

    print('='*32)
    print('Coef LASSO: ', lasso_model.coef_)
    print('Coef RIDGE: ', ridge_model.coef_)