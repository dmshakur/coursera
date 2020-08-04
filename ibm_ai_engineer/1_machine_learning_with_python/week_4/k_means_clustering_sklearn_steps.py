
# Import libraries

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from mpl_toolkits.plot3d import axes3D

# Initialize x and y with random data

x, y = make_blobs(n_samples = 5000, centers = [[4, 4], [-2, -1], [2, -3], [1, 1]])

np.random.seed(0)

# Create model

k_means = KMeans(init = 'k-means++', n_clusters = 4, n_init = 12)

# Fit model to x

k_means.fit(x)

# Grab the labels for each point in the model using k-means' .labels_attribute and save it

k_means_labels = k_means.labels_

# Grab the cluster centroid locations

k_means_cluster_centers = k_means.cluster_centers_

# Create a visual plot for the model

fig = plt.figure(figsize = (6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
ax = fig.add_subplot(1, 1, 1)

for k, col in zip(range(len([4, 4], [-2, -1], [2, -3], [1, 1]]), colors):
    my_members = (k_means_labels == k)
    cluster_center = k_means_cluster_centers[k]
    ax.plot(x[my_members, 0], x[my_members, 1], 'w', markerfacecolor = col, marker = '.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor = col, markerredgecolor = 'k', markersize = 6)
ax.set_title('K-Means')
ax.set_xticks(())
ax.set_yticks(())
plt.show()