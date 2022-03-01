import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def data_split(dataframe, label):
    X = dataframe.drop(label, axis=1)
    y = dataframe[label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=77)

    return X_train, X_test, y_train, y_test


def model_train(X_train, X_test, y_train, y_test):
    # Instantiation a knn model
    knn_class = KNeighborsClassifier()
    knn_class.fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)
    
    print('='*64)
    print(f'KNN acurracy: {accuracy_score(y_test, knn_pred)}')
    
    # Instantiation a bagging model
    bag_class = BaggingClassifier(base_estimator=knn_class, n_estimators=50)
    bag_class.fit(X_train, y_train)
    bag_pred = bag_class.predict(X_test)

    print('='*64)
    print(f'Bagging acurracy: {accuracy_score(y_test, bag_pred)}')

    print('='*64)

def main():
    # Loading the data
    df_heart = pd.read_csv('./data/heart.csv')

    X_train, X_test, y_train, y_test = data_split(df_heart, 'target')

    model_train(X_train, X_test, y_train, y_test)
    

if __name__ == '__main__':
    main()