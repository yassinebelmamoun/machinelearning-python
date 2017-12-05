# Machine Learning Implementation with Python 

Implementation of relevant machine learning algorithms in a clear and concise way

## Synopsis

Machine learning is currently a buzzword. The goal here is to walk away with a better understanding of 8 different machine learning algorithms.
You will find in this repository the implementation of relevant machine learning algorithms in a clear and concise way.

## Summary

1. Installation
2. Algorithms
  * Linear Regression
  * Logistic Regression
  * Dicision Trees
  * Support vector Machines (SVM)
  * K-Nearest Neighbors
  * Random Forests
  * K-Means Clustering

## Installation & Dependencies

I use Python 3.6.2.

We will use the following libraries:
  * Pandas
  * Matplotlib
  * Numpy
  * Seaborn
You can simply install requirements.txt by doing:
```
pip install -r requirements.txt
```

## Dataset

We will use the following data for our tests:

XXX

## Algorithms

### Linear Regression
Linear regression is a supervised learning algorithm that predicts an outcome based on continuous features.
Linear regression offers the ability to be run on a single variable (simple linear regression) or on many features (multiple linear regression).
The way it works is by assigning optimal weights to the variables in order to create a line (ax + b) that will be used to predict output.

```python
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model

df = pd.read_csv('dataset.csv')
df.columns = ['X', 'Y']
print(df)

# Visualization
sns.set_context('notebook', font_scale=1.1)
sns.set_style('ticks')
sns.lmplot('X','Y', data=df)
plt.ylabel('Response')
plt.xlabel('Explanatory')
plt.show()

# Implementation
linear = linear_model.LinearRegression()
trainX = np.asarray(df.X[20:len(df.X)]).reshape(-1, 1)
trainY = np.asarray(df.Y[20:len(df.Y)]).reshape(-1, 1)
testX = np.asarray(df.X[:20]).reshape(-1, 1)
testY = np.asarray(df.Y[:20]).reshape(-1, 1)
linear.fit(trainX, trainY)
linear.score(trainX, trainY)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print('R² Value: \n', linear.score(trainX, trainY))
predicted = linear.predict(testX)
```

### Logistic Regression

Logistic regression is a supervised classification algorithm and therefore is useful for estimating discrete values.

It is often used for predicting the probability of an event using the logistic function in order to get an output between 0 and 1.

Some of the underlying aspects of logistic regression come up in many other important machine learning algorithms like neural networks.

```python

```
### Decision Trees


Decision trees are a form of supervised learning that can be used for both classification and regression purposes.

They are typically utilized for classification purposes.

The model takes in an instance and then goes down the tree, testing significant features against a determined conditional statement.

Depending on the result, it will go down to the left or right child branch and onward after that. Typically the most significant features in the process will fall closer to the root of the tree.

Decision trees are becoming increasingly popular and can serve as a strong learning algorithm for any data scientist to have in their repertoire, especially when coupled with techniques like random forests, boosting, and bagging.


### Support vector machines

Support vector machines, also known as SVM, are a well-known supervised classification algorithm that create a dividing line between different categories of data.

The way this vector is calculated, in simple terms, is by optimizing the line so that the closest point in each of the groups will be farthest away from each other.

This vector is by default and often visualized as being linear, however this doesn’t have to always be the case. The vector can take a nonlinear form as well if the kernel type is changed from the default type of ‘gaussian’ or linear.


### K-Nearest Neighbors

K-Nearest Neighbors, KNN for short, is a supervised learning algorithm specializing in classification.

The algorithm looks at different centroids and compares distance using some sort of function (usually Euclidean), then analyzes those results and assigns each point to the group so that it is optimized to be placed with all the closest points to it.


### Random forests

Random forests are a popular supervised ensemble learning algorithm.

‘Ensemble’ means that it takes a bunch of ‘weak learners’ and has them work together to form one strong predictor.

In this case, the weak learners are all randomly implemented decision trees that are brought together to form the strong predictor — a random forest.


### K-Means Clustering

K-Means is a popular unsupervised learning classification algorithm typically used to address the clustering problem.

The ‘K’ refers to the user inputted number of clusters.

The algorithm begins with randomly selected points and then optimizes the clusters using a distance formula to find the best grouping of data points. It is ultimately up to the data scientist to select the correct ‘K’ value.


### Principal Components Analysis

PCA is a dimensionality reduction algorithm that can do a couple of things for data scientists.

Most importantly, it can dramatically reduce the computational footprint of a model when dealing with hundreds or thousands of different features.

It is unsupervised, however the user should still analyze the results and make sure they are keeping 95% or so of the original dataset’s behavior.

There’s a lot more to address with PCA.
