import pandas as pd

from sklearn.cluster import MiniBatchKMeans


def model_train(X):
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8)
    kmeans.fit(X)

    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.predict(X)

    print('='*64)
    print(f'Total centroids: {len(cluster_centers)}')
    
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