
# Machine Learning With Python Week 1

> Introduction to machine learning: In this week, you will learn about applications of Machine Learning in different fields such as health care, banking, telecommunication, and so on. You’ll get a general overview of Machine Learning topics such as supervised vs unsupervised learning, and the usage of each algorithm. Also, you understand the advantage of using Python libraries for implementing Machine Learning models.
>
>Key Concepts
>* To give examples of Machine Learning
>* To demonstrate the Python libraries for Machine Learning
>* To classify Supervised vs. Unsupervised algorithms


### Introduction to machine learning

#### An example of a problem to be solved by machine learning, cancer
In contrast with a benign tumor, a malignant tumor is a tumor that may invade its surrounding tissue or spread around the body, and diagnosing it early might be the key to a patient’s survival.

Characteristics from benign or malignant tumors seem be widely different.

Once data has been preprocessed it can be used to iteratively go through your model with the final goal of testing it on a test data-set.

> ***What is machine learning*** Machine learning, is the subfield of computer science that gives "computers the ability to learn without being explicitly programmed." - Arthur Samuel
>
> American pioneer in the field of computer gaming and artificial intelligence, coined the term "machine learning" in 1959 while at IBM.

#### How machine learning works
Data -> Feature extraction -> A machine learning model -> predictive output

Feature extraction starts from an initial set of measured data and builds derived values (features) intended to be informative and non-redundant, facilitating the subsequent learning and generalization steps, and in some cases leading to better human interpretations. Feature extraction is related to dimensionality reduction.

#### Examples of machine learning

Any kind of suggestion service such as Google ads, Netflix and Amazon recommendation systems.

#### Major machine learning techniques

* Regression / Estimations
    * Predicting continuous values
* Classification
    * Predicting the item class/category of a case
* Clustering
    * Finding the structure of data; summarization
* Associations
    * Associating frequent co-occuring items / events
* Anomaly detection
    * Discovering abnormal and unusual cases
* Sequence mining
    * Predicting next events; click-stream (Markov model, HMM)
* Dimension reduction
    * Reducing the size of data (PCA)
* Recommendation systems
    * Recommending systems

#### Difference between, artificial intelligence, machine learning and deep learning

* A.I. components:
    * Computer vision
    * Language processing
    * Creativity
    * Etc.
* Machine learning:
    * Classification
    * Clustering
    * Neural networks
* Revolution in ML:
    * Deep learning, 

### Python for machine learning

#### Python libraries for ML:

* Numpy
    * Math library to work with n-dimensional arrays in python
* Scipy
    * Signal processing, optimization, statistics and scientific and high performance computation
* Matplotlib
    * A visualization package for displaying models and other information
* Pandas
    * Data structures. Data importing, manipulation, and analysis. Manipulating numerical tables and time-series
* Scikit Learn
    * Collection of algorithms and tools used for machine learning
    * Popular, free, classification - regression - clustering, works with Numpy and Scipy
    * Most of the tasks that are in a machine learning pipeline are in Scikit Learn including: Data preprocessing, train / test split, algorithm setup, model fitting, prediction, evaluation, model export

#### Example of Scikit Learn

```python
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

from sklearn import svm
clf = svm.SVC(gamma = 0.001, C = 100.)

clf.fit(X_train, y_train)

clf.predict(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, yhat, labels = [1, 0]))

import pickle
s = pickle.dumps(clf)
```

### Supervised vs unsupervised

#### What is supervised learning

We "teach the model" by training it with some data from a labeled data set.

In order to train data you need to know the class or result of a row of data.

You can have two kinds of data, numerical or categorical.

There are two types of supervised learning techniques regression and classification.

With regression you can predict a data of a continuous nature

#### What is unsupervised learning
The model works on its own to discover information, and has more sophistication.
##### Unsupervised learning techniques
* Dimension reduction
    * Reducing redundant features to make the classification easier
* Density estimation
    * Explores data to find some structure within it
* Market basket analysis
    * Based upon the theory if your going to buy a certain group of items you're going to buy another group of items
* Clustering
    * Grouping data points or objects that are somehow similar
    * Discovering structure
    * Summarization
    * Anomaly Detection
##### Supervised Learning
* Classification: Classifies and labels data
* Regression: Predicts trends using previous labeled data
* Has more evaluation methods than unsupervised machine learning
* Controlled environment
##### Unsupervised learning
* Clustering: Finds patterns and groupings from unlabeled data
* Has fewer evaluation methods than supervised learning
* Less controlled environment