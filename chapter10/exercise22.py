# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from scipy.cluster.hierarchy import cut_tree, linkage, dendrogram


if __name__ == '__main__':
    x, y = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1, center_box=(-10.0, 10.0), shuffle=True, random_state=1)
    tree = linkage(x, method="ward", metric="euclidean")
    plt.figure(dpi=300)
    dendrogram(tree, labels=np.arange(0, len(x)))
    plt.hlines(y=10, linestyles=":", xmin=0, xmax=x.shape[0] * 10)
    plt.xticks(rotation=90, fontsize=2)
    plt.show()
    labels = cut_tree(tree, height=10)
    print(np.unique(labels.ravel()).shape[0])
