# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import DBSCAN


if __name__ == '__main__':
    x, y = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1, center_box=(-10.0, 10.0), shuffle=True, random_state=1)
    model = DBSCAN(eps=1.5, min_samples=3, n_jobs=-1)
    model.fit(x)
    print(np.unique(model.labels_).shape[0])