
# Import libraries
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import basemap
from pylab import rcParams
import sklearn.utils
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Data generation
def create_data_points(centroid_location, num_samples, cluster_deviation):
    x, y = make_blobs(n_samples = num_samples, centers = centroid_location, cluster_std = cluster_deviation)
    x = StandardScaler().fit_transform(x)
    return x, y

# Initialize x and y
x, y = create_data_points([[4, 3], [2, -1], [-1, 4]], 1500, 0.5)

# Modeling
epsilon = 0.3
minimum_samples = 7
dbscan = DBSCAN(eps = epsilon, min_samples = minimum_samples).fit(x)
labels = dbscan.labels_

# Distinguish outliers, creating an array of booleans  using the dbscan labels
core_samples_mask = np.zeros_like(dbscan.labels_, dtype = bool)
core_samples_mask[dbscan.core_samples_indices_] = True

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# remove repetition in labels by turning it into a set
unique_lables = set(labels)

# Data visualization
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_lables)))
for k, col in zip(unique_lables, colors):
    if k == 1:
        col = 'k'
    class_member_mask = (labels == k)
    xy = x[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], s = 50, c = [col], marker = u'o', alpha = 0.5)
    xy = x[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], s = 50, c = [col], marker = u'o', alpha = 0.5)


# Using DBSCAN with csv files
!wget -O weather-stations20140101-20141231.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/weather-stations20140101-20141231.csv
raw_data = pd.read_csv('weather-stations20140101-20141231.csv')

# Cleaning data
data = raw_data[pd.notnull(data['Tm'])]
data = data.reset_index(drop = True)

# Visualization with basemap
rcParams['figure.figsize'] = (14,10)
llon=-140
ulon=-50
llat=40
ulat=65
pdf = pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) & (pdf['Lat'] > llat) &(pdf['Lat'] < ulat)]
my_map = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)
my_map.drawcoastlines()
my_map.drawcountries()
# my_map.drawmapboundary()
my_map.fillcontinents(color = 'white', alpha = 0.3)
my_map.shadedrelief()
# To collect data based on stations        
xs,ys = my_map(np.asarray(pdf.Long), np.asarray(pdf.Lat))
pdf['xm']= xs.tolist()
pdf['ym'] =ys.tolist()
#Visualization1
for index,row in pdf.iterrows():
#   x,y = my_map(row.Long, row.Lat)
   my_map.plot(row.xm, row.ym,markerfacecolor =([1,0,0]),  marker='o', markersize= 5, alpha = 0.75)
#plt.text(x,y,stn)
plt.show()

sklearn.utils.check_random_state(1000)
clus_dataset = data[['xm', 'ym']]
clus_dataset = np.nan_to_num(clus_dataset)
clus_dataset = StandardScaler().fit_transform(clus_dataset)

# Compute DBSCAN
db  = DBSCAN(eps = 0.15, min_samples = 10).fit(clus_dataset)
core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
data['clus_db'] = labels
real_cluster_num = len(set(labels)) - (1 if -1 in labels else 0)
cluster_num = len(set(labels))

# Visualization of clusters based on location
rcParams['figure.figsize'] = (14, 10)
my_map = Basemap(projection = 'merc', resolution = 'l', area_thresh = 1000.0, llcrnrlon = llon, llcrnrlat = llat, urcrnlon = ulon, urcrnrlat = ulat)
my_map.drawcoastlines()
my_map.drawcountries()
my_map.fillcontinents(color = 'white', alpha = 0.3)
colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))
for clust_number in set(labels):
    c = (([0.4, 0.4, 0.4]) if clust_Number == -1 else colors[np.int(clust_number)])
    clust_set = data[data.clus_db == clust_number]
    my_map.scatter(clust_set.xm, clust_set.ym, color = c, marker = 'o', s = 20, alpha = 0.85)
    if clust_number != -1:
        cenx = np.mean(clust_set.xm)
        ceny = np.mean(clust_set.ym)
        plt.text(cenx, ceny, str(clust_number), fontsize = 25, color = 'red')
        print('cluster ' + str(clust_set) + ', avg temp: ' + str(np.mean(clust_set.Tm)))

# Clustering stations based on their location, mean, max and min temperature
sklearn.utils.check_random_state(1000)
clus_dataset = data[['xm', 'ym', 'Tx', 'Tm', 'Tn']]
clus_dataset = np.nan_to_num(clus_dataset)
clus_dataset = StandardScaler().fit_transform(clus_dataset)

# Compute DBSCAN
db = DBSCAN(eps = 0.3, min_samples = 10).fit(clus_dataset)
core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
data['clus_db'] = labels
real_cluster_num = len(set(labels)) - (1 if -1 in labels else 0)
cluster_num = len(set(labels))

# Visualization of clusters based on location and temperature
rcParams['figure.figsize'] = (14, 10)
my_map = Basemap(projection = 'merc', resolution = 'l', area_thresh = 1000.0, llcrnrlon = llon, llcrnrlat = llat, urcrnrlon = olon, urcrnrlat = ulat)
my_map.drawcountries()
my_map.drawcountries()
my_map.fillcontinents()
my_map.shadedrelief()
colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, cluster_num)
for clust_number in set(labels):
    c = (([0.4, 0.4, 0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = data[data.clus_db == clust_number]
    my_map.scatter(clust_set.xm, clust_set.ym, color = c, marker = 'o', s = 20, alpha = 0.85)
    if clust_number != -1:
        cenx = np.mean(clust_set.xm)
        ceny = np.mea(clust_set.ym)
        plt.text(cenx, ceny, str(clust_number), fontsize = 25, color = 'red')
        print('cluster ' + str(clust_number) + ', avg temp: ' + str(np.mean(clust_set.Tm)))