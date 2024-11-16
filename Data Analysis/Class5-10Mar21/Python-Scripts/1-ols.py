#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 21:23:36 2021

@author: franciscocantuortiz
"""

# https://www.geeksforgeeks.org/linear-regression-python-implementation/

# Exercise No 1

import numpy as np 
import matplotlib.pyplot as plt 
  
def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1) 
  
def plot_regression_line(x, y, b): 
    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "m", 
               marker = "o", s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "g") 
  
    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    # function to show plot 
    plt.show() 
  
def main(): 
    # observations 
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]) 
  
    # estimating coefficients 
    b = estimate_coef(x, y) 
    print("Estimated coefficients:\nb_0 = {}  \
          \nb_1 = {}".format(b[0], b[1])) 
  
    # plotting regression line 
    plot_regression_line(x, y, b) 
  
if __name__ == "__main__": 
    main() 
    

# Exercise No 2
    
# https://realpython.com/linear-regression-in-python/#advanced-linear-regression-with-statsmodels

# Linear Regression With statsmodels
# You can implement linear regression in Python relatively easily by using the package statsmodels as well. Typically, this is desirable when there is a need for more detailed results.

# The procedure is similar to that of scikit-learn.

# Step 1: Import packages

# First you need to do some imports. In addition to numpy, you need to import statsmodels.api:

import numpy as np
import statsmodels.api as sm
# Now you have the packages you need.

# Step 2: Provide data and transform inputs

# You can provide the inputs and outputs the same way as you did when you were using scikit-learn:

x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)
# The input and output arrays are created, but the job is not done yet.

# You need to add the column of ones to the inputs if you want statsmodels to
# calculate the intercept 𝑏₀. It doesn’t takes 𝑏₀ into account by default. 
# This is just one function call:

x = sm.add_constant(x)
# That’s how you add the column of ones to x with add_constant(). It takes the input array x as an argument and returns a new array with the column of ones inserted at the beginning. This is how x and y look now:

print(x)

print(y)

# You can see that the modified x has three columns: the first column of ones (corresponding to 𝑏₀ and replacing the intercept) as well as two columns of the original features.

# Step 3: Create a model and fit it

# The regression model based on ordinary least squares is an instance of the class statsmodels.regression.linear_model.OLS. This is how you can obtain one:

model = sm.OLS(y, x)
# You should be careful here! Please, notice that the first argument is the output, followed with the input. There are several more optional parameters.

# To find more information about this class, please visit the official documentation page.

# Once your model is created, you can apply .fit() on it:

results = model.fit()
# By calling .fit(), you obtain the variable results, which is an instance of the class statsmodels.regression.linear_model.RegressionResultsWrapper. This object holds a lot of information about the regression model.

# Step 4: Get results

# The variable results refers to the object that contains detailed information about the results of linear regression. Explaining them is far beyond the scope of this article, but you’ll learn here how to extract them.

# You can call .summary() to get the table with the results of linear regression:

print(results.summary())


# Exercise No 3


from sklearn import datasets ## imports datasets from scikit-learn
data = datasets.load_boston() ## loads Boston dataset from datasets library 

print(data.DESCR)

import numpy as np
import pandas as pd
# define the data/predictors as the pre-set feature names  
df = pd.DataFrame(data.data, columns=data.feature_names)

df.head()

def get_df_size(df, header='Dataset dimensions'):
  print(header,
        '\n# Attributes: ', df.shape[1], 
        '\n# Entries: ', df.shape[0],'\n')
  
get_df_size(df)

df.info()

df.columns

df.describe()

df.dtypes

# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])

## Without a constant

import statsmodels.api as sm

X = df["RM"]
y = target["MEDV"]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()

# Add a constant (intercept b0)
import statsmodels.api as sm # import statsmodels 

X = df["RM"] ## X usually means our input variables (or independent variables)
y = target["MEDV"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
model.summary()

# Multiple Linear Regression
X = df[['RM', 'LSTAT']]
y = target['MEDV']
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
model = sm.OLS(y, X).fit()
predictions = model.predict(X)
model.summary()
