# -*- coding: utf-8 -*-
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth


if __name__ == '__main__':
    x, y = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1, center_box=(-10.0, 10.0), shuffle=True, random_state=1)
    model = MeanShift(
        bandwidth=estimate_bandwidth(x, quantile=0.2, n_samples=200),
        bin_seeding=True,
        n_jobs=-1
    )
    model.fit(x)
    print(model.cluster_centers_)