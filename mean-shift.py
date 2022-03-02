import pandas as pd

from sklearn.cluster import MeanShift


def model_train(X):
    meanshift = MeanShift()
    meanshift.fit(X)

    cluster_centers = len(meanshift.cluster_centers_)
    labels = meanshift.labels_

    print('='*64)
    print(f'Total centroids: {cluster_centers}')
    
    return labels


def main():
    # Loading the data
    df = pd.read_csv('./data/candy.csv')
    
    X = df.drop('competitorname', axis=1)
    labels = model_train(X)

    df['categories'] = labels

    return df


if __name__ == '__main__':
    df = main()
    print(df.head())