#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 22:55:40 2021

@author: franciscocantuortiz
"""
# https://realpython.com/logistic-regression-python/

# Logistic Regression

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# For the purpose of this example, let’s just create arrays for the input (𝑥) 
# and output (𝑦) values:

# numpy.arange() creates an array of consecutive, equally-spaced values within
#  a given range
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# The array x is required to be two-dimensional. It should have one column 
# for each input, and the number of rows should be equal to the number of 
# observations. To make x two-dimensional, you apply .reshape() with 
# the arguments -1 to get as many rows as needed and 1 to get one column

# x has two dimensions:
# One column for a single input
# Ten rows, each corresponding to one observation
# y is one-dimensional with ten items. Again, each item corresponds to one 
# observation. It contains only zeros and ones since this is a binary
# classification problem.
print(x)
print(y)

# Once you have the input and output prepared, you can create and define your 
# classification model. You’re going to represent it with an instance of the 
# class LogisticRegression:
model = LogisticRegression(solver='liblinear', random_state=0)

# The above statement creates an instance of LogisticRegression and binds 
# its references to the variable model. LogisticRegression has several 
# optional parameters that define the behavior of the model and approach

# Once the model is created, you need to fit (or train) it. 
# Model fitting is the process of determining the coefficients 𝑏₀, 𝑏₁, …, 𝑏ᵣ 
# that correspond to the best value of the cost function. 
# You fit the model with .fit():

model.fit(x, y)

# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#                    intercept_scaling=1, l1_ratio=None, max_iter=100,
#                    multi_class='warn', n_jobs=None, penalty='l2',
#                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
#                    warm_start=False)

# Alternatively:
model = LogisticRegression(solver='liblinear', random_state=0).fit(x, y)

# You can quickly get the attributes of your model. For example, 
# the attribute .classes_ represents the array of distinct values that y takes:

model.classes_

# You can also get the value of the slope 𝑏₁ and the intercept 𝑏₀ of the 
# linear function 𝑓 like so:

model.intercept_
model.coef_

# Step 4: Evaluate the Model
# Once a model is defined, you can check its performance with 
# .predict_proba(), which returns the matrix of probabilities that 
# the predicted output is equal to zero or one:
model.predict_proba(x)

# n the matrix above, each row corresponds to a single observation. 
# The first column is the probability of the predicted output being zero, 
# that is 1 - 𝑝(𝑥). The second column is the probability that the output 
# is one, or 𝑝(𝑥).


# You can get the actual predictions, based on the probability matrix and 
# the values of 𝑝(𝑥), with .predict():
model.predict(x)

# When you have nine out of ten observations classified correctly, 
# the accuracy of your model is equal to 9/10=0.9, 
# which you can obtain with .score():

model.score(x, y)

# You can get more information on the accuracy of the model with a 
# confusion matrix. In the case of binary classification, 
# the confusion matrix shows the numbers of the following:

# True negatives in the upper-left position
# False negatives in the lower-left position
# False positives in the upper-right position
# True positives in the lower-right position
# To create the confusion matrix, you can use confusion_matrix() and provide 
# the actual and predicted outputs as the arguments:

confusion_matrix(y, model.predict(x))

cm = confusion_matrix(y, model.predict(x))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


# You can get a more comprehensive report on the classification with
#  classification_report():

print(classification_report(y, model.predict(x)))



# Improve the Model
# You can improve your model by setting different parameters. 
# For example, let’s work with the regularization strength C equal to 10.0, 
# instead of the default value of 1.0:

model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
model.fit(x, y)

model.intercept_

model.coef_

model.predict_proba(x)

model.score(x, y)

confusion_matrix(y, model.predict(x))

print(classification_report(y, model.predict(x)))

