
# Import relevant libraries

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Create pandas data frame from csv file

raw_data = pd.read_csv('file_name_here')

# Create numpy array for X and Y

x = raw_data[['all_columns_except_y_here']].values
y = raw_data['y_column_here'].values

# Preprocess X

x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))

# train test split

TEST_SIZE = # Float for percentage of data to be used for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = TEST_SIZE)

# Train and create the model

k = # Whatever number you want to use for k
knn = KNeighborsClassifier(n_neighbors = k).fit(x_train, y_train)

# Predict the values for the test set

y_hat = knn.predict(x_test)

# Evaluate accuracy

print('Train set accuracy: ', metrics.accuracy_score(y_train, knn.predict(x_train)))
print('Test set accuracy: ', metrics.accuracy_score(y_test, y_hat))

# Evaluating the best number for k
# While calculating the mean accuracy and the standard accuracy

k = 10 # Should be a high number so you can explore a range of possible values
mean_acc = np.zeros((k - 1))
std_acc = np.zeros((k - 1))
confusion_matrix = []
for n in range(1, k):
    knn_of_n = KNeighborsClassifier(n_neighbors = n).fit(x_train, y_train)
    y_hat_of_n = knn_of_n.predict(x_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, y_hat_of_n)
    std_acc[n - 1] = np.std(y_hat == y_test) / np.sqrt(y_hat_of_n.shape[0])

# Plotting model accuracy for 1 - n of knn

plt.plot(range(1, k), mean_acc, 'g')
plt.fill_between(range(1, k), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha = 0.1)
plt.legend(('Accuracy', '+/- 3xstd'))
plt.ylabel('Accuracy')
plt.xlabel('Number of nighbors (K)')
plt.tight_layout()
plt.show()