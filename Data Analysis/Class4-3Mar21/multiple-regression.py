#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 18:32:31 2021

@author: franciscocantuortiz
"""
# https://realpython.com/linear-regression-in-python/#multiple-linear-regression

# Multiple Linear Regression With scikit-learn
# You can implement multiple linear regression following the same steps as 
# you would for simple regression.

# Steps 1 and 2: Import packages and classes, and provide data

# First, you import numpy and sklearn.linear_model.LinearRegression and 
# provide known inputs and output:

import numpy as np
from sklearn.linear_model import LinearRegression

x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

print(x)
print(y)

# Step 3: Create a model and fit it

# The next step is to create the regression model as an instance of 
# LinearRegression and fit it with .fit():

model = LinearRegression().fit(x, y)

# The result of this statement is the variable model referring to the object 
# of type LinearRegression. It represents the regression model fitted with 
# existing data.

# Step 4: Get results

# You can obtain the properties of the model the same way as in the case of 
# simple linear regression:

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

print('intercept:', model.intercept_)

print('slope:', model.coef_)


# You obtain the value of 𝑅² using .score() and the values of the estimators 
# of regression coefficients with .intercept_ and .coef_. Again, .intercept_ 
# holds the bias 𝑏₀, while now .coef_ is an array containing 𝑏₁ and 𝑏₂ 
# respectively.

# In this example, the intercept is approximately 5.52, and this is the value 
# of the predicted response when 𝑥₁ = 𝑥₂ = 0. The increase of 𝑥₁ by 1 yields the rise of the predicted response by 0.45. Similarly, when 𝑥₂ grows by 1, the response rises by 0.26.

# Step 5: Predict response

# Predictions also work the same way as in the case of simple linear 
# regression:

y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')

y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1)
print('predicted response:', y_pred, sep='\n')

# You can predict the output values by multiplying each column of the input 
# with the appropriate weight, summing the results and adding the intercept 
# to the sum.

# You can apply this model to new data as well:

x_new = np.arange(10).reshape((-1, 2))
print(x_new)

y_new = model.predict(x_new)
print(y_new)
