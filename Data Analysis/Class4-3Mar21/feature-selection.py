#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:25:40 2021

@author: franciscocantuortiz
"""

# Feature Selection
# https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b

#importing libraries
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
#Loading the dataset
x = load_boston()
df = pd.DataFrame(x.data, columns = x.feature_names)
df["MEDV"] = x.target
X = df.drop("MEDV",1)   #Feature Matrix
y = df["MEDV"]          #Target Variable
df.head()

# ----------------------------------------------------------------------

# Filter Method...

# Using Pearson Correlation as criteria for selecting features with high correlation

plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#Correlation with output variable
cor_target = abs(cor["MEDV"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features

# Check for independence of predictor variables
print(df[["LSTAT","PTRATIO"]].corr())
print(df[["RM","LSTAT"]].corr())

# From the above code, it is seen that the variables RM and LSTAT are highly 
# correlated with each other (-0.613808). Hence we would keep only one 
# variable and drop the other. We will keep LSTAT since its correlation 
# with MEDV is higher than that of RM.
# After dropping RM, we are left with two feature, LSTAT and PTRATIO. 
# These are the final features given by Pearson correlation.

# ----------------------------------------------------------------------

# Wrapper Method
# A wrapper method needs one machine learning algorithm and uses its 
# performance as evaluation criteria. This means, you feed the features to 
# the selected Machine Learning algorithm and based on the model performance 
# you add/remove the features. This is an iterative and computationally 
# expensive process but it is more accurate than the filter method.
# There are different wrapper methods such as Backward Elimination, the
# Forward Selection, Bidirectional Elimination and RFE. We will discuss 
# Backward Elimination and RFE here.

# Backward Elimination
# As the name suggest, we feed all the possible features to the model at 
# first. We check the performance of the model and then iteratively remove 
# the worst performing features one by one till the overall performance of 
# the model comes in acceptable range.
# The performance metric used here to evaluate feature performance is pvalue. 
# If the pvalue is above 0.05 then we remove the feature, else we keep it.

# # Here we are using OLS model which stands for “Ordinary Least Squares”. 
# This model is used for performing linear regression.

#Adding constant column of ones, mandatory for sm.OLS model
X_1 = sm.add_constant(X)

#Fitting sm.OLS model
model = sm.OLS(y,X_1).fit()
model.pvalues

# As we can see that the variable ‘AGE’ has highest pvalue of 0.9582293 which
# is greater than 0.05. Hence we will remove this feature and build the model 
# once again. This is an iterative process and can be performed at once with 
# the help of loop. This approach is implemented below, which would give the 
# final set of variables which are CRIM, ZN, CHAS, NOX, RM, DIS, RAD, TAX, 
# PTRATIO, B and LSTAT

#Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

# RFE
# The Recursive Feature Elimination (RFE) method works by recursively 
# removing attributes and building a model on those attributes that remain. 
# It uses accuracy metric to rank the feature according to their importance. The RFE method takes the model to be used and the number of required features as input. It then gives the ranking of all the variables, 1 being most important. It also gives its support, True being relevant feature and False being irrelevant feature.
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 7)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  

#Fitting the data to model
model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)

# Here we took LinearRegression model with 7 features and RFE gave feature 
# ranking as above, but the selection of number ‘7’ was random. Now we need 
# to find the optimum number of features, for which the accuracy is the 
# highest. We do that by using loop starting with 1 feature and going up to 
# 13. We then take the one for which the accuracy is highest.

#no of features
nof_list=np.arange(1,13)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

# As seen from above code, the optimum number of features is 10. We now feed 
# 10 as number of features to RFE and get the final set of features given by
# RFE method, as follows:

cols = list(X.columns)
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 10)             
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  

#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)

# -----------------------------------------------------------------

# Embedded methods are iterative in a sense that takes care of each iteration
#  of the model training process and carefully extract those features which 
# contribute the most to the training for a particular iteration. 
# Regularization methods are the most commonly used embedded methods which 
# penalize a feature given a coefficient threshold.
# Here we will do feature selection using Lasso regularization. 
# If the feature is irrelevant, lasso penalizes it’s coefficient and make 
# it 0. Hence the features with coefficient = 0 are removed and the rest 
# are taken.
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
# Image for post
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
# Image for post
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")

# Here Lasso model has taken all the features except NOX, CHAS and INDUS.

# Conclusion:
# We saw how to select features using multiple methods for Numeric Data and 
# compared their results. Now there arises a confusion of which method to 
# choose in what situation. Following points will help you make this decision.
# Filter method is less accurate. It is great while doing EDA, it can also 
# be used for checking multi co-linearity in data.
# Wrapper and Embedded methods give more accurate results but as they are
# computationally expensive, these method are suited when you have lesser 
# features (~20).
