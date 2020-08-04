
# Machine Learning With Python Week 3

> In this week, you will learn about classification technique. You practice with different classification algorithms, such as KNN, Decision Trees, Logistic Regression and SVM. Also, you learn about pros and cons of each method, and different classification accuracy metrics.
>
>Key concepts:
> * To understand different Classification methods.
> * To apply Classification algorithms on various data-sets to solve real world problems. To understand evaluation methods in Classification.

## Classification
### What is classification
* A supervised learning approach
* Categorizing some unknown items into a discrete set of categories or 'classes'
* The target attribute is a categorical variable with discrete values

###  How does classification
You can create a classifier with either binary or multi-class data.

### Classification use cases
* Which category a customer belongs to
* Whether a customer switches to another provider/brand?
* Whether a customer responds to a particular advertising campaign?

### Types of classification algorithms in machine learning
* Decision tress (ID3, C4.5, C5.0)
* Naive bayes
* Linear discriminant analysis
* K-nearest neighbor
* Logistic regression
* Neural networks
* Support vector machines

## K-nearest neighbor (KNN)

### What is k-nearest neighbors?
* A method for classifying cases based on their similarity to other cases
* Cases that are near each other are said to be 'neighbors'
* Based on similar cases with same class labels are near each other

### The k-nearest neighbors algorithm
1. Pick a value for k.
2. Calculate the distance of unknown case from all cases
3. Select the k-observations in the training data that are nearest to the unknown data point
4. Predict the response of the unknown data point using the most popular response value from the k-nearest neighbors

### What is the best value of k for knn?
K and k-nearest neighbors is the number of nearest neighbors to examine.
Capturing the noise of the data is when you look at the nearest neighbor of a point and it belongs to a class that is not majorly represented. A low value of k, results in a highly complex model.

You want a general model that can work for any data.

If you choose a high value of k, such as 20, it will be over-generalized.

Computing continuous targets using knn, knn can also be used for regression

### Evaluation metrics in classification

Evaluation metrics explain the performance of the model. When we want to find the accuracy we compare y to y^.

### Jaccard index
> Also known as the jaccard similarity coefficient

We can define jaccard as the size of the intersection divided by the size of the union of two label sets.

---

#### Jaccard equation:
$$
J(y, \hat{y}) = \frac{|A\cap B|}{|A\cup B|}
$$

---

### F1-score
#### Confusion matrix
The columns in a confusion matrix show the predicted labels by the classifier and the rows show the actual values.

The right column in the top row is the amount of false negatives, or type 2 errors, the left column bottom row is the amount of false positives, or type 1 errors.

Precision equation:
$$ TP / (TP + FP) $$
Recall equation:
$$ TP / (TP + FN) $$

#### F1-score
The f1-score is the harmonic average of the precision and recall.

F1 score equation:
$$ 2 * (Precision * Recall) / (Precision + Recall) $$

The f1-score is best at 1 and worst at 0.

You can average f1-scores with a confusion matrix

### Log loss
Performance of a classifier where the predicted output is a probability value between 0 and 1.

You can calculate the log loss for each row in a data-set using the log loss equation.

Log loss equation:
$$ (y * log(\hat{y}) + (1 - y) * log(1 - \hat{y})) $$

The result is how far the prediction is from the actual label, a model with a lower log loss has better accuracy.

## Decision trees

### What is a decision tree
The basic intuition behind a decision tree is to map out all possible decision paths in the form of a tree.

Decision trees are built by splitting the training set into distinct nodes, where one node contains all of or most of one category of the data.

Decision trees start with the most general categories and then work their way down to ever more specific categories.

* Each internal node corresponds to a test
* Each branch corresponds to a result of the test
* Each leaf node assigns a something to a class

### Decision tree learning algorithm
1. Choose an attribute from your data-set
1. Calculate the significance of an attribute in splitting the data
1. Split data base on the value of the best attribute
1. Then go to each branch and repeat the previous 3 steps for the rest of the attributes

### Building decision trees
How do we build a decision tree based on a data-set?

Decision trees are built using recursive partitioning to classify the data.

### What attribute is best
Categories split into different versions of those categories. If we cannot say with high confidence that either category in the split is a strong determiner of the subclass then that attribute is poorly selected.

#### When selecting the best attribute you want:
* More predictiveness
* Less impurity
* Lower entropy
We are looking for a decrease in impurity of nodes after splitting them up based on that feature.

When we create a tree we want to have the least amount of entropy we can muster.

#### Entropy
* Measure of randomness or uncertainty
* The lower the entropy, the less uniform the distribution, the purer the node
* The amount of entropy in a node is the amount of random data in that node, and is calculated for each node
If the samples are completely homogenous the entropy is 0, if the samples are equally divided then the entropy is 1.

0 = Perfect

1 = Worst

---

p = proportion_or_ratio_of_category

#### Entropy per node equation:
$$
 entropy = p(A)log(p(A)) - p(B)log(p(B))
$$

---

You can calculate the impurity of the target data before splitting.

The best attribute to choose is the tree with the highest information gain after splitting

### What is information gain
Information gain is the information that can increase the level of certainty after splitting.

Information gain equation: entropy_before_split - weighted_entropy_after_split
$$ entropy\,before\,split - weighted\,entropy\,after\,split $$

As the amount of randomness or entropy decreases the amount of certainty or information gain increases.

## Logistic regression

### What is logistic regression
Logistic regression is a classification algorithm for categorical variables.

### Logistic regression applications
* Predicting the probability of a person having a heart attack
* Predicting the mortality in injured patients
* Predicting a customers propensity to purchase a product or halt a subscription
* Predicting the probability of failure of a given process or product
* Predicting the likelihood of a homeowner defaulting on a mortgage

### When is logistic regression suitable?
* If your data is binary
    * 0/1, yes/no, true, false
* If you need probabilistic results
* When you need a linear decision boundary
* If you need to understand the impact of the feature

### Logistic regression training

#### General cost function
$$
\sigma (\theta^TX) \longrightarrow P(y=1|x)
$$

* Change the weight -> reduce the cost
* Cost function

Usually the square of the below equation is used because of the possibility of a negative result.
$$ 
Cost(\hat{y}, y) = \frac{1}{2}(\sigma(\theta^TX) - y)^2 
$$

$$
J(\theta) = \frac{1}{m}\displaystyle\sum_{i-1}^m Cost(\hat{y}, y)
$$

We find the best weights or parameters that minimize the cost function by calculating the minimum point of the cost function and it will show the best parameters for the model. You can find the minimum point of a function using the derivative of a function. Which is difficult to calculate.

$$
Cost(\hat{y}, y) = \begin{cases} -log(\hat{y}) \text{ if } y = 1 \\ -log(\hat{1 - \hat{y}}) \text{ if } y = 0 \end{cases}
$$

#### Total cost function:
$$
J(\theta) = \frac{1}{m}\displaystyle\sum_{i-1}^m y^i log(\hat{y}^i) + (1 - y^i)log(1 - \hat{y}^i)
$$

### Minimizing the cost function of the model
* How to find the best parameters for our model
    * Minimize the cost function
* How to minimize the cost function
    * Using gradient descent
* What is gradient descent
    * An iterative approach to finding the minimum of a function
    * A technique to use the derivative of a cost function to change the parameter values, in order to minimize the cost

### Training algorithm recap
1. Initialize the parameters randomly.
$$ \theta^T = [\theta_0, \theta_1, \theta_2, \ldots] $$
2. Feed the cost function with training data and calculate the error.
$$ J(\theta) $$
3. Calculate the gradient of the cost function.
$$
\nabla J = [\frac{\partial j}{\partial\theta_1},\frac{\partial j}{\partial\theta_2},\frac{\partial j}{\partial\theta_3},\ldots,\frac{\partial j}{\partial\theta_k}]
$$
4. Update the weights with new values.
$$ \theta_{new} = \theta_{previous} - \eta\nabla J $$
5. Go to step 2 until cost is small enough.
6. Make a prediction with new data.
$$ P(y = 1 | x) = \sigma(\theta^TX) $$

## Support vector machine (SVM)

You can use support vector machines as a classifier.

### What is a supervised algorithm that classifies cases by finding a separator.
1. Mapping data to a high-dimensional feature space
2. Finding a separator

### Data transformation

If we have data that is not linearly separable then we can increase the dimensions of the space it is in.

Mapping data into a higher dimensional space is called kerneling.
#### Kerneling:
* Linear
* Polynomial
* RBF (Radial bases function)
* Sigmoid
Each characteristic has it's own pros and cons

#### Kernel function/equation:
$$ \phi(x) = [x, x^2] $$

### Using SVM to find the hyperplane
SVMs are based on the idea of finding a hyperplane that best divides a data-set into two classes.

In a two dimensional space, you can think of a hyperplane as a line that linearly separates the each class. The best hyperplane is the line between two classes that has the highest margin, or space that separates each class from the line.

Points closest to the hyperplane are called support vectors. Only support vectors matter for our goal.

### Pros and cons of SVM
* Advantages:
    * Accurate in high-dimensional spaces
    * Memory efficient
* Disadvantages:
    * Prone for over-fitting
    * No probability estimation
    * Not very efficient computationally, only good for small data-sets

### SVM application
* Image recognition
* Text category assignment
* Detecting spam
* Sentiment analysis
* Gene expression classification
* Regression, outlier detection, and clustering