#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:17:30 2021

@author: franciscocantuortiz
"""

# https://realpython.com/linear-regression-in-python/

# Regression Analysis 
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

print(x)
print(y)

model = LinearRegression()

model.fit(x, y)

# Or:
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

print('intercept:', model.intercept_)
print('slope:', model.coef_)

# Or:
new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print('intercept:', new_model.intercept_)
intercept: [5.63333333]
print('slope:', new_model.coef_)

# Prediction:
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')

z = np.array([6, 16, 26, 36, 46, 56]).reshape((-1, 1))
y_pred = model.predict(z)
print('predicted response:', y_pred, sep='\n')

x_new = np.arange(5).reshape((-1, 1))
print(x_new)
y_new = model.predict(x_new)
print(y_new)

