from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

x = np.linspace(0.2,1,100)

y = 0.8*x + np.random.randn(100)*0.1
X = np.vstack([x, y]).T
np.random.shuffle(X)
print(X)
print(type(X))
print(X.shape)

pca = PCA(n_components=2)
pca.fit(X)
PCA(copy=True, n_components=2, whiten=False)

