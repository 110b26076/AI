%matplotlib inline
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples = 300, centers = 3, cluster_std = 1)

plt.figure(figsize = (6, 6))
plt.scatter(X[:, 0], X[:, 1], s = 30, marker = "x")
plt.show()

kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)
Y = kmeans.predict(X)

plt.figure(figsize = (6, 6))
plt.scatter(X[:, 0], X[:, 1], c = Y, s = 30, marker = "x", cmap = "plasma")
centers = kmeans.cluster_centers_ 
plt.scatter(centers[:, 0], centers[:, 1], c = "black", s = 200, alpha = 0.8)
plt.show()
