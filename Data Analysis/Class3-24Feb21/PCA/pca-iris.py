#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 08:52:42 2021

@author: franciscocantuortiz
"""

# Principal Component Analysis (PCA) using the IRIS dataset
# For data visualization

# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60 


# Load Libraries
import pandas as pd

# Read the IRIS dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

#Print the first 5 rows 
df.head()

#last 5 rows
df.tail(5)

#Head and tail
print(df)

#a.	Get the dimensions of the data.
df.shape

#Get the summary.
df.describe()

# Sepal attribute
df['sepal length'].head()

# Petal attribute
df['petal length'].head()

#d.	Explore the data types of each column
df.dtypes

# PCA is effected by scale so you need to scale the features in 
# your data before applying PCA. Use StandardScaler to help you 
# standardize the dataset’s features onto unit scale 
# (mean = 0 and variance = 1) which is a requirement for the 
#optimal performance of many machine learning algorithms

from sklearn.preprocessing import StandardScaler
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

# Reduce dataset from four to two dimensions
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
finalDf

# Visualize 2D projection
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

# Explain variance
pca.explained_variance_ratio_
