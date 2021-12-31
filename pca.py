import pandas as pd
# import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

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

    print(X_train.shape)
    print(y_train.shape)

    # configurando el PCA
    pca = PCA(n_components=3)  # n_componenest por defecto = min(n_registros, n_columnas)
    pca.fit(X_train, y_train)

    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train, y_train)

    # obervación gráfica del aporte de variables de PCA
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()

    # entrenando el modelo
    logistic = LogisticRegression(solver='lbfgs')

    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print('Score PCA: ', logistic.score(dt_test, y_test))

    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(dt_train, y_train)
    print('Score IPCA: ', logistic.score(dt_test, y_test))