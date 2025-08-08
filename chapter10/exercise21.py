# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AffinityPropagation


if __name__ == '__main__':
    x, y = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1, center_box=(-10.0, 10.0), shuffle=True, random_state=1)
    model = AffinityPropagation(preference=-50)
    model.fit(x)
    print(model.cluster_centers_)