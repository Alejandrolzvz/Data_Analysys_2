#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 19:25:58 2021

@author: franciscocantuortiz
"""

# https://medium.com/the-code-monster/split-a-dataset-into-train-and-test-datasets-using-sk-learn-acc7fd1802e0

import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Load the car prices dataset
df = pd.read_csv('carprices.csv')

# Print first rows
df.head()

# Plot Mileage Versus Sell Price
plt.scatter(df['Mileage'],df['SellPrice'])

# Define predictors 
X = df[['Mileage','Age']]

# Define Target Variable (Response)
Y = df['SellPrice']

# Print the first rows of predictor variables
X.head(10)

# Print the first rows of target variable
Y.head(10)

# Separate train and test data (Hold-out)
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size =0.2)

# print the train data
x_train
y_train

# print the test data
x_test
y_test

#Apply Multiple Linear Regression OLS
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
import statsmodels.api as sm

# Fit the equation to data
clf.fit(x_train,y_train)

# Predict the price on test data
clf.predict(x_test)

clf.score(x_test,y_test)

# Summary with Statsmodels
import statsmodels.api as sm
model = sm.OLS(y_test, x_test).fit()
predictions = model.predict(X)
model.summary()