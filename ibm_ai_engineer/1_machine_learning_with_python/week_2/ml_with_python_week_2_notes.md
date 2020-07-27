
# Machine Learning With Python Week 1

> Regression: In this week, you will get a brief intro to regression. You learn about Linear, Non-linear, Simple and Multiple regression, and their applications. You apply all these methods on two different data-sets, in the lab part. Also, you learn how to evaluate your regression model, and calculate its accuracy.
>
>Key concepts:
> * To understand the basics of regression
> * To apply Simple and Multiple, Linear and Non-Linear Regression on a data-set for estimation.

# Linear Regression

## Introduction to regression

We can use regression to predict continuous values.

In regression there are two types of variables the dependent variable (Y), and the independent variable (X).

Our regression relates Y to a function of X.

### Two types of regression models

* Simple regression: Using one independent variable
    * Simple linear regression
    * simple non-linear regression
* Multiple regression: Using more than one independent variable
    * Multiple linear regression
    * Multiple non-linear regression

### Applications of regressions

* Sales forecasting
* Satisfaction analysis
* Price estimation
* Employment income

### Regression algorithms

* Ordinal regression
* Poisson regression
* Fast forest quantile regression
* Linear, polynomial, lasso, stepwise, ridge
* Bayesian linear regression
* Neural network regression
* Decision forest regression
* boosted decision tree regression
* K nearest neighbor (KNN)

## Simple linear regression

### The simple linear regression equation
 
$$ \hat{y} = \theta_0 + \theta_1 * X_1 $$

y(hat) is the predicted value, is dependent of X[1], is what we are trying to predict

Theta_0 is known as the intercept

Theta_1 is known as the slope or gradient of the fitting line

X_1 is the predictor

We must calculate both theta[0] and theta[1] to find the line that is the best fit for the data.

### How to find the best fit

The error is also called the residual error.

The mean of the residual error can be an indicator of how well the model performs.

The objective of linear regression is to find the best parameters for both theta[0] and theta[1].

You can use a mathematical formula to find theta[0:1]

Theta[0] is also called the bias coefficient

### Pros of linear regression

* Very fast
* No parameter tuning
* Easy to understand, and highly interpretable

## Model evaluation in regression models
Our goal is to actively predict an unknown case.

### Model evaluation approaches
* Train and test on the same data-set
* Train / Test split

### Best approach for accurate results
One solution is to select a portion of our data-set to train the model. Then use the rest as a test set.

The labels are called actual values, which we need to predict. When we train the model and test it we compare the actual values (y) of the test set to the predicted values (y hat).

### Training and testing on the same data-set
You train your model on the whole data-set then test on a portion of the data-set.

#### Training accuracy
* Training accuracy
    * High training accuracy isn't necessarily a good thing
    * result of over-fitting
        * Over-fit: The model is overly trained to the data-set, which may capture noise and produce a non-generalized model
* Out-of-sample accuracy
    * It's important that our models have a high, out-of-sample accuracy
    * How can we improve out-of-sample accuracy

### Train / test split approach
Train/test split:
* Mutually exclusive
* More accurate evaluation on out-of-sample accuracy
* Make sure you train your model on the testing set after you finish testing as you don't want to lose out on that valuable information
* Highly dependent on which data-sets the data is trained and tested

### How to use k-fold cross-validation
If we have k equals 4 folds we split the data up into 4 *folds*
* 1st fold: Used for testing, and the rest for training
* 2nd fold: Used for testing, and the rest for training
* 3rd fold: Used for testing, and the rest for training
* 4th fold: Used for testing, and the rest for training

Then the accuracy is calculated with the mean of accuracy of k folds

## Evaluation metrics in regression models

Evaluation metrics are used to explain the performance of the model, which provide a key role in the development of a model, as it provides insight to areas that require improvement.

### What is an error of the model?
#### MAE: Mean absolute error is the mean of all the errors

#### MSE: Mean squared error is the mean of the squared errors. It is more popular than mean absolute error because the focus is geared more towards large errors.

#### RMSE: Root mean squared error is the square root of the mean squared error. This is one of the most popular of the evaluation metrics because root mean squared error is interpretable in the same units as the response vector or y units, making it easy to relate its information.

#### RAE: Relative absolute error or the residual sum of square, where y(bar) is a mean value of y, takes the total absolute error and normalizes it by dividing the total absolute error of the simple predictor. 

#### RSE: Relative squared error is very similar to relative absolute error but is widely adopted by the data science community, as it is used for calculating r-squared.

> R-Squared is not an error per se, but is a popular metric for the accuracy of your model. It represents how close the data values are to the fitted regression line. The higher the r-squared, the better the model fits your data.
$$ R^2 = 1 - RSE $$

## Multiple linear regression

Using two or more independent variables is multiple linear regression

### Examples of linear regression
* Independent variables effectiveness on prediction
    * For example: Does revision time, test anxiety, lecture attendance, and gender have any effect on the exam performance of students
* Predicting impacts of changes
    * For example: If we were reviewing a persons health data a multiple linear regression can tell you how much that person's blood pressure go up (or down) for every unit increase (or decrease) in the BMI of a patient

### Predicting continuous values with multiple linear regression
As with simple linear regression multiple linear regression is trying to predict a continuous value.
You can find out how much each feature impacts the output.

Multiple linear regression can be shown as a dot product of two vectors, the parameters vector and the feature set vector. In the vector X the first input is equal to 1 so when it is multiplied by theta[0] it retains its value.

### Using MSE to expose the errors in the model

The residual error is equal to:
$$
(y - \hat{y})
$$

The mean of all residual errors shows us how bad our model is performing on the data-set: MSE.

The best model for our data-set is the one with the smallest error.

### Estimating multiple linear regression parameters

#### How to estimate theta
* Ordinary least squares tries to estimate the values of the coefficients by minimizing the mean square error.
    * linear algebra operations to optimize the value for the theta
    * Takes a long time for large data-sets (10k+ rows), for greater values try other faster options
* An optimization algorithm
    * Gradient descent, starts optimization with random values for each coefficient
    * Proper approach if you have a very large data-set

### Making predictions with multiple linear regression

If the coefficient is higher that means it is likely to have a larger impact on the data-set.

### Concerns about multiple linear regression

* How to determine whether to use simple or multiple linear regression?
* How many independent variables should we use for our prediction? 
    * Adding too many independent variables may result in an over fit model. Over fitting creates weights not generalized enough to create meaningful predictions in the real world.
* Should the independent variable be continuous?
    * Categorical independent variables can be incorporated into a regression model by converting them into numerical variables. For example, given a binary variable such as car type, the code dummy zero for manual and dummy one for automatic cars
* What are the linear relationships between the dependent variable and the independent variable?
    * There are a number of ways to check for linear relationships, like with scatter plots. If the relationship is non-linear you need to use multiple linear regression.

## Non-Linear Regression

### Should we use linear regression
If the data shows a curved trend you won't get accurate results with linear regression.
Different regressions are best fit to what the data movement looks like.

### What is polynomial regression
* Some curvy data can be modeled by a polynomial regression
* A polynomial regression model can be transformed into a linear regression model.

### What is non-linear regression
* To model non-linear relationship between the dependent variable and a set of independent variables.
* y(hat) must be a non-linear function of the parameters theta, not necessarily the features x. We cannot use the OLS method and estimation of the parameters is not easy.

### Linear vs non-linear regression
* How can I know if a problem is linear or non-linear in as easy way?
    * Inspect visually, you can calculate the correlation coefficient between all independent variables and dependent variables
    * Based on accuracy
* How should I model my data, if it displays non-linear on a scatter plot?
    * Polynomial regression
    * Non-linear regression model
    * Transform your data
