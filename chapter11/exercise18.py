# -*- coding: utf-8 -*-
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans


if __name__ == '__main__':
    x, y = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1, center_box=(-10.0, 10.0), shuffle=True, random_state=1)
    model = KMeans(n_jobs=-1)
    for i in range(2, 5):
        model.n_clusters = i
        model.fit(x)
        print(i, model.score(x))
