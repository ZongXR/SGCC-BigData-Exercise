# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.decomposition import PCA


if __name__ == '__main__':
    data = pd.read_csv("./9.4/transformer_overload.csv")
    model = PCA(n_components=0.98, svd_solver="full")
    x = data.iloc[:, -3:]
    model.fit(x)
    print(model.n_components_)