
# Machine Learning With Python Week 4

> In this section, you will learn about different clustering approaches. You learn how to use clustering for customer segmentation, grouping same vehicles, and also clustering of weather stations. You understand 3 main types of clustering, including Partitioned-based Clustering, Hierarchical Clustering, and Density-based Clustering.
>
>Key Concepts
>* To understand different types of clustering algorithms.
>* To apply clustering on different types of data-sets.

## Intro to clustering
Customer segmentation is the practice of partitioning a customer base into groups of individuals that have similar characteristics.

Customer segmentation is one of the popular use cases for clustering. 

### What is clustering
Finding clusters in a data-set unsupervised
#### What is a cluster
A group of objects that are similar to other objects in the cluster, and dissimilar to data points in other clusters.

### Clustering vs classifications
Generally speaking classification is a supervised learning technique where each training data instance belongs to a particular class, that when training we are aware of what class they belong to. With clustering however, the data is unlabeled and the process is unsupervised.

### Clustering applications
* Retail/marketing:
    * Identifying buying patterns of various customer groups
    * Recommending new books or movies to new customers
* Banking:
    * Fraud detection in credit card use
    * Identifying clusters of customers, loyal vs churn
* Insurance:
    * Fraud detection in claims analysis
    * Insurance risk of customers
* Publication:
    * Auto-categorizing news based on their content
    * Recommending similar news articles
* Medicine:
    * Characterizing patient behavior
* Biology:
    * Clustering is used to group genes with similar expressions or markers that may have family ties

### Why clustering
* Exploratory data analysis
* Summary generation
* Outlier detection
* Finding duplicates
* Pre-processing step

### Clustering algorithms
* Partitioned-based clustering
    * Relatively efficient
    * E.g. k-means, k-median, fuzzy c-means
* Hierarchical clustering
    * Produces trees of clusters
    * E.g. Agglomerative, Divisive
* Density-based clustering
    * Produces arbitrary shaped clusters
    * E.g. DBSCAN

## K-means clustering

### K-means algorithms
* Partitioning clusters
* K-means divides the data into non-overlapping subsets (clusters) without any cluster-internal structure
* Examples within a cluster are very similar
* Examples across different clusters are very different

### Determine the similarity or dissimilarity
The objective of k-means is to form clusters in such a way that similar samples go into a cluster and dissimilar samples fall into different clusters. Conventionally we use the distance of samples from each other is used to shape the clusters.

### Dimensional similarity/distance
#### Equation:
$$
Dis(x_1, x_2) = \sqrt{\textstyle\sum_{i=0}^n(x_{1i} - x_{2i})^2}
$$

### K-means clustering steps
The first step is to decided on the number of clusters.
1. Initialize centroids = k randomly.
    * There are two approaches to choose these centroids. One we can randomly choose k observations out of the data-set and use these observations as the initial means. Two, we can create k random points as centroids of the clusters. 
2. Distance calculation
    * We have to calculate the distance of each data point to each cluster. Where you will form a matrix comprising of those distances. 
    * The goal is to minimize the distance to all data points in it's cluster and maximize the distance from other cluster's data points. 
    * Euclidean distance is used to measure the distance from the data point to the centroid. You can use other types Euclidean is just the most popular.
#### The equation for SSE, sum of squared errors, that is used to find the distances between each point and its centroid:
$$ SSE = \textstyle\sum_1^n(x_i - C_j)^2 $$
3. Assign each point to the closest centroid
4. Compute the new centroids for each cluster
    * The centroid of each of the three clusters becomes the new mean. This continues until the centroid no longer moves.
5. Repeat until there are no changes
    * You need to complete steps 2-4 until the centroids will move no more. The result may be a local optimum. Usually very fast.

### K-means accuracy
* External approach
    * Compare the clusters with the ground truth, if it is available. Because k-means is an unsupervised algorithm we usually don't have ground truth in real world problems.
* Internal Approach
    * Average the distance between data points within a 
    
### Choosing k
The correct choice of k is often ambiguous. As it is very dependent on the shape and scale of the distribution of points in a data-set.

One of the more common ways to figure this out is to iterate through the different values of k to find where we minimize the error.

Increasing k will always decrease the error. So the value of the metric as a function of k is plotted and the elbow point is determined where the rate of decrease sharply shifts. This is called the elbow method.

## Hierarchical clustering
Hierarchical clustering algorithms build a hierarchy of clusters where each node is a cluster that is consisting of the clusters of its daughter nodes.

#### There are two types:
* Agglomerative
    * Bottom Up
    * Each observation start in its own cluster and pairs of clusters merge together to from a hierarchy
    * Agglomeration means to amass or collect things
    * More popular among data scientists
* Divisive:
    * Top Down
    * You start with all observations in a cluster, then create more specific clusters as you go

### Agglomerative clustering
Every data point starts out with its own cluster. Then you create parent clusters with clusters with similarities or relations.

If clustering data on a map one way to map the distance of a cluster to the other points is to use the center of the cluster.

A common way to merge clusters is to take the two clusters that have the smallest difference in to merge, and repeat until done.

It is typically visualized as a dendrogram.

### Agglomerative algorithm
1. Create n clusters, one for each data point
2. Compute the distance proximity matrix
3. Repeat:
    1. Merge the two closest clusters,
    2. Update the proximity matrix
4. Stop after reaching the specified number of clusters or if only one cluster remains

### Distance between clusters
* Single-linkage clustering
    * Minimum distance between clusters
* Complete-linkage clustering
    * Maximum distance between clusters
* Average linkage clustering
    *Average distance between clusters
* Centroid linkage clusters
    * Distance between cluster centroids

### Advantages vs disadvantages of hierarchical clustering
#### Advantages
* Doesn't require number of clusters to be specified
* Easy to implement
* Produces a dendrogram, which helps with understanding data
#### DIsadvantages
* Can never undo any previous steps throughout algorithm
* Generally long run times
* Sometimes difficult to identify the number of clusters by the dendrogram

### Hierarchical clustering vs k-means clustering
#### k-means clustering
* Much more efficient
* Requires the number of clusters to be specified
* Gives only one partitioning of the data based on the predefined number of clusters
* Potentially returns different clusters each time it is run due to random initialization centroids
#### Hierarchical clustering
* Can be slow for large data-sets
* Does not require the number of clusters to run
* Gives more than one partitioning depending on the resolution
* Always generates the same clusters

## Density-based clustering
A density based clustering algorithm which is appropriate to use when examining spatial data.

### K-means vs density based clustering
* k-means assigns all points to a cluster even if they do not belong to any

* Density based clustering locates regions of high density, and separates outliers

### DBSCAN for class identification
DBSCAN a very popular density based clustering method is effective for tasks like class identification on a spatial context.

### What is DBSCAN
* DBSCAN (density based spatial clustering of applications with noise)
    * Is one of the most common clustering algorithms
    * Works based on defsity of objects
* R (Radius of neighborhood)
    * Radis (R) that if includes enough number of points within, we call it a dense area
* M (Min number of neighbors)
    * The minimum number of data points we want in a neighborhood to define a cluster

### How DBSCAN works
Each point is either:
* Core point
    * Within our neighborhood of the point there are at least n points
* Border point
    * A data point is a border point if A: it's neighborhood contains less than m data points or B: it is reachable from some core point. Here reachability means it is within our distance from a core point.
* Outlier point
    * A point that is not a core point and is not close enough to be reachable from a core point

The whole point of the algorithm is to look at each point and find its type first then group points afterwards.

A cluster is formed if it has at least one fore point and all reachable core points plus all their border points.

### Advantages of DBSCAN
* Arbitrarily shaped clusters
* Robust to outliers
* Does not require specification of the number of clusters
