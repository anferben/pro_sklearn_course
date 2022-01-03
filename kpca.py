import pandas as pd
# import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # loading the data
    df_heart = pd.read_csv('./data/heart.csv')

    print(df_heart.head())

    # separamos features de target values
    df_features = df_heart.drop('target', axis=1)
    df_target = df_heart['target']

    # estandarizando los datos
    dt_features = StandardScaler().fit_transform(df_features)

    # separación del dataset
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target,
        test_size=0.3, random_state=42
    )

    kpca = KernelPCA(n_components=4, kernel='poly')
    kpca.fit(X_train)

    dt_train = kpca.transform(X_train)
    dt_test = kpca.transform(X_test)

    logistic = LogisticRegression(solver='lbfgs', max_iter=100)

    logistic.fit(dt_train, y_train)
    print('SCORE KPCA: ', logistic.score(dt_test, y_test))