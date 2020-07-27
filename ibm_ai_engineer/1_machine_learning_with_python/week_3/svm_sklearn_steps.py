
# Import libraries

import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import itertools
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, jaccard_similarity_score, f1_score

# Load csv data

raw_data = pd.read_csv('file_path')

# Convert non numerical data to numbers and preprocess data

data =  raw_data[pd.to_numeric(raw_data['non_numeric_data'], errors = 'coerce').notnull()]
data['non_numeric_data'] = data['non_numeric_data'].astype('int')

feature_data = data[['all_column_names_except_y']]

data['y_column'] = data['y_column'].astype('int')
y = np.asarray(data['y_column'])

# Create x variable

x = np.asarray(feature_data)

# Split data-set into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)

# Create SVM

model = svm.SVC(kernel = 'rbf')
model.fit(x_train, y_train)

# Predict with test set

y_hat = model.predict(x_test)

# Evaluate the accuracy of the model, creating the confusion matrix function

def plot_confusion_matrix(cm, classes, normalize = False, title = 'confusion matrix', cmap = plt.cm.Blue):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment = 'center', color = 'white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, y_hat, labels = [2, 4])
np.set_printoptions(precision = 2)
print(classification_report(y_test, y_hat))
plt.figure()
plot_confusion_matrix(cnf_matrix, classes = ['y_one', 'y_two'], normalize = False, title = 'Confusion matrix')

# Get the f1 score

f1 = f1_score(y_test, y_hat, average = 'weighted')

# Get the jaccard index

jaccard = jaccard_similarity_score(y_test, y_hat)

