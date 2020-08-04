# Import libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg

# Load data, in this case a csv file
raw_data = pd.read_csv('file_name_here')

# converting data frame into numpy array
x = raw_data[['column_values_here']].values

# Converting categorical variables into numerical representations, dummy variables
# Do this for all the categories that need dummy variables
labelencoder_categorical_variable_name = preprocessing.LabelEncoder()
labelencoder_categorical_variable_name.fit(['categories_here'])
x[:, column_number_here] = labelencoder_categorical_variable_name.transform(x[:, column_Number_here])

# Create y
y = raw_data['column_name_for_y_here']

# Splitting the data into a training set and a testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 3)

# Create and instance of the DecisionTreeClassifier
# the argument criterion = 'entropy' will enable seeing the information gain of each node
dec_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 4)

# fitting the model with the training data
dec_tree.fit(x_train, y_train)

# Make predictions with the x_test and store it in a variable
pred_tree = dec_tree.predict(x_test)

# Compute and display the accuracy of the model
print('Decision tree\'s accuracy', metrics.accuracy_score(y_test, pred_tree))

# Visualization
dot_data = StringIO()
file_name = 'file_name_here'
feature_names = raw_data.columns[0: last_feature_name_here]
targets_names = raw_data['y_column_here'].unique().tolist()
out = export_graphviz(dec_tree, 
    feature_names = feature_names, 
    out_file = dot_data, 
    class_names = np.unique(y_train), 
    filled = True, 
    special_characters = True, 
    rotate = False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(file_name)
img = mpimg.imread(file_name)
plt.figure(figsize = (100, 200))
plt.imshow(img, interpolation = 'nearest')