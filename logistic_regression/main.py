import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.linear_model import LogisticRegression

df = pd.read_csv('iris.csv')
df.columns = ['X1', 'X2', 'X3', 'X4', 'Y']

#Visualisation
sns.set_context('notebook', font_scale=1.1)
sns.set_style('ticks')
sns.regplot('X1','Y', data=df, logistic=True)
plt.ylabel('Probability')
plt.xlabel('Explanatory')

# Implementation
logistic = LogisticRegression()
X = (np.asarray(df.X)).reshape(-1, 1)
Y = (np.asarray(df.Y)).ravel()
logistic.fit(X, Y)
logistic.score(X, Y)
print('Coefficient: \n', logistic.coef_)
print('Intercept: \n', logistic.intercept_)
print('R² Value: \n', logistic.score(X, Y))
