
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

# Load csv file
raw_data = pd.read_csv('file_path')

# Drop null values
data[['column_names']] = raw_data[['column_names']].apply(pd.to_numeric, errors='coerce')
data = data.dropna()
data = data.reset_index(drop = True)

# Select the feature set
feature_set = data[['column_names']]

# Normalize data
x = feature_set.values
min_max_scaler = MinMaxScaler()
feature_matrix = min_max_scaler.fit_transform(x)

# Clustering with scipy
leng = feature_matrix.shape[0]
d = scipy.zeros([leng, leng])
for i in range(leng):
    for j in range(leng):
        d[i, j] = scipy.spatial.distance.euclidean(feature_matrix[i], feature_matrix[j])

# Create z
z = hierarchy.linkage(d, 'complete')

# Cutting clusters
max_d = 3
clusters = fcluster(z, max_d, criterion = 'distance')

# Determine the number of clusters
k = 5
clusters = fcluster(x, k, criterion = 'maxclust')

# Visualize the dendrogram
fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (data['manufact'][id], data['model'][id], int(float(data['type'][id])) )
    
# Create the dendrogram
dendro = hierarchy.dendrogram(z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')

# Clustering using sci-kit learn
dist_matrix = distance_matrix(feature_matrix, feature_matrix)

# Creating the model
agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(feature_matrix)

# Add a new row to the dataframe to show it's cluster
data['cluster_'] = agglom.labels_

# Display with a scatter plot
n_clusters = max(agglom.labels_) + 1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
clusters_labels = list(range(0, n_clusters))
plt.figure(figsize = (16, 14))
for color, label in zip(colors, clusters_labels):
    subset = data[data.cluster_ == label]
    for i in subset.index:
        plt.text(subset.horsepow[i], subset.mpg[i], str(subset['model'][i]), rotation = 25)
    plt.scatter(subset.horsepow, subset.mpg, s = subset.price * 10, c = color, label = 'cluster' + str(label), alpha = 0.5)
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()

# Calculate the number of cases for each cluster
cases = data.groupby(['cluster_', ''])['cluster_'].count()

# Characteristics of each cluster
agg_ = data.groupby(['cluster_', ''])['category_names'].mean()

#
