
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import scipy, pylab
import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import fcluster
import matplotlib.cm as cm

# Generating random data
x1, y1 = make_blobs(n_samples = 50, centers = [[4, 4], [-2, -1, [1, 1], [10, 4]], cluster_std = 0.9)

# Plot data 
plt.scatter(x1[:, 0], x1[:, 1], marker = 'o')

# Create and fit the model
agglomerative = AgglomerativeClustering(n_clusters = 4, linkage = 'average')
agglomerative.fit(x1, y1)

# Plot the clustered data
plt.figure(figsize = (6, 4))
x_min, x_max = np.min(x1, axis = 0), np.max(x1, axis = 0)
x1 = (x1 - x_min) / (x_max - x_min)
for i in range(x1.shape[0]):
    plt.text(x1[i, 0], x1[i, 1], str(y1[i]), color = plt.cm.nipy_spectral(agglomerative.labels_[i] / 10.), fontdict = {'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.scatter(x1[:, 0], x1[:, 1], marker = '.')
plt.show()

# Creating a dendrogram
dist_matrix = distance_matrix(x1, x1)

# Create z
z = hierarchy.linkage(distance_matrix, 'complete')

# Create the dendrogram
dendro = hierarchy.dendrogram(z)

