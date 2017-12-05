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

[I'm an inline-style link](https://www.google.com)

## Algorithms

### Linear Regression
Linear regression is a supervised learning algorithm that predicts an outcome based on continuous features.
Linear regression offers the ability to be run on a single variable (simple linear regression) or on many features (multiple linear regression).
The way it works is by assigning optimal weights to the variables in order to create a line (ax + b) that will be used to predict output.

```python
from sklearn import linear_model
df = pd.read_csv(‘linear_regression_df.csv’)
df.columns = [‘X’, ‘Y’]
df.head()
```

### Logistic Regression
### Decision Trees
### Support vector machines
### K-Nearest Neighbors
### Random forests
### K-Means Clustering
