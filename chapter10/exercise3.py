# -*- coding: utf-8 -*-
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier


if __name__ == '__main__':
    centers = [
        [-2, 2],
        [2, 2],
        [0, 4]
    ]
    x, y = make_blobs(n_samples=60, centers=centers, random_state=0, cluster_std=0.60)
    model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    model.fit(x, y)
    print(model.predict([
        [1, 3]
    ]))
