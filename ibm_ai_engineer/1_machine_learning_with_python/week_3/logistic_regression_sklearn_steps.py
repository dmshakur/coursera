
# Import libraries
import itertools
import pandas as pd
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, log_loss, jaccard_similarity_score

# Import csv file as pandas data frame
raw_data = pd.read_csv('file_path')

# Convert y value to integer
raw_data['y_value_column'] = raw_data['y_value_column'].astype('int')

# Define x and y
x = np.asanyarray(raw_data[['all_columns_except_y']])

y = np.asarray(raw_data['y_column_here'])

# Normalize the data set
x = preprocessing.StandardScaler().fit(x).transform(x)

# train/test split the data-set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)

# creating and fitting the model with the training set
log_reg = LogisticRegression(C = 0.01, solver = 'liblinear').fit(x_train, y_train)

# Predict with the test set 
y_hat = log_reg.predict(x_test)

# Get the estimates for all classes ordered by the label of classes
y_hat_prob = log_reg.predict_proba(x_test)

# Evaluation: Jaccard index
jaccard_index = jaccard_similarity_score(y_test, y_hat)

# Confusion matrix
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
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
    plt.xlabel('predicted label')
print(confusion_matrix(y_test, y_hat, labels = [1, 0]))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_hat, labels = [1, 0])
np.set_printoptions(precision = 2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes = ['class_name = 1', 'class_name = 0'], normalize = False, title = 'Confusion matrix')

# Display: precision, recall, f1-score and support
print(classification_report(y_test, y_hat))

# Getting the log loss
log_reg_loss = log_loss(y_test, y_hat_prob)
