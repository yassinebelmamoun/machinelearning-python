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
### Support vector machines
### K-Nearest Neighbors
### Random forests
### K-Means Clustering
