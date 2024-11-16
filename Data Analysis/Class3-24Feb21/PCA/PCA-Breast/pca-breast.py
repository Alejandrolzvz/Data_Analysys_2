#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:26:06 2021

@author: franciscocantuortiz
"""

#Principal Component Analysis (PCA) with Brest Cancer dataset

# https://www.datacamp.com/community/tutorials/principal-component-analysis-in-python

#Get the dataset from sklearn datasets
from sklearn.datasets import load_breast_cancer

# Separate data
breast = load_breast_cancer()
breast_data = breast.data

#Get shape
breast_data.shape

# Although we will not need the labels, let us take a look
breast_labels = breast.target
breast_labels.shape

# Now you will import numpy since you will be reshaping the 
# breast_labels to concatenate it with the breast_data so that 
# you can finally create a DataFrame which will have both the 
# data and labels.
import numpy as np
labels = np.reshape(breast_labels,(569,1))

# After reshaping the labels, you will concatenate the data 
# and labels along the second axis, which means the final 
# shape of the array will be 569 x 31.
final_breast_data = np.concatenate([breast_data,labels],axis=1)
final_breast_data.shape

# Now you will import pandas to create the DataFrame of the 
# final data to represent the data in a tabular fashion.
import pandas as pd
breast_dataset = pd.DataFrame(final_breast_data)

# Let's quickly print the features that are there in the 
# breast cancer dataset!
features = breast.feature_names
features

# If you note in the features array, the label field is 
# missing. Hence, you will have to manually add it to the 
# features array since you will be equating this array with 
# the column names of your breast_dataset dataframe.
features_labels = np.append(features,'label')

# Great! Now you will embed the column names to the 
# breast_dataset dataframe.
breast_dataset.columns = features_labels

# Let's print the first few rows of the dataframe.
breast_dataset.head()

# Since the original labels are in 0,1 format, you will 
# change the labels to benign and malignant using .replace 
# function. You will use inplace=True which will modify the 
# dataframe breast_dataset.
breast_dataset['label'].replace(0, 'Benign',inplace=True)
breast_dataset['label'].replace(1, 'Malignant',inplace=True)

# Let's print the last few rows of the breast_dataset.
breast_dataset.tail()

# Visualizing the Breast Cancer data
# You start by Standardizing the data since PCA's output is 
# influenced based on the scale of the features of the data.
# It is a common practice to normalize your data before 
# feeding it to any machine learning algorithm.

# To apply normalization, you will import StandardScaler 
# module from the sklearn library and select only the features
# from the breast_dataset you created in the Data Exploration 
#step. Once you have the features, you will then apply scaling
# by doing fit_transform on the feature data.

# While applying StandardScaler, each feature of your data 
# should be normally distributed such that it will scale the 
# distribution to a mean of zero and a standard deviation of 
# one.

from sklearn.preprocessing import StandardScaler
x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x) # normalizing the features
x.shape

# Let's check whether the normalized data has a mean of zero 
# and a standard deviation of one.
np.mean(x),np.std(x)

# Let's convert the normalized features into a tabular 
# format with the help of DataFrame.
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_breast = pd.DataFrame(x,columns=feat_cols)
normalised_breast.tail()

# Now comes the critical part, the next few lines of code 
# will be projecting the thirty-dimensional Breast Cancer 
# data to two-dimensional principal components.

# You will use the sklearn library to import the PCA module, 
# and in the PCA method, you will pass the number of 
# components (n_components=2) and finally call fit_transform 
# on the aggregate data. Here, several components represent 
# the lower dimension in which you will project your higher 
#dimension data.
from sklearn.decomposition import PCA
pca_breast = PCA(n_components=2)
principalComponents_breast = pca_breast.fit_transform(x)
# Next, let's create a DataFrame that will have the principal
#  component values for all 569 samples.
principal_breast_Df = pd.DataFrame(data = principalComponents_breast
             , columns = ['principal component 1', 'principal component 2'])
principal_breast_Df.tail()

# Once you have the principal components, you can find the 
# explained_variance_ratio. It will provide you with the 
# amount of information or variance each principal component 
# holds after projecting the data to a lower dimensional 
# subspace.
print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))

# From the above output, you can observe that the principal 
# component 1 holds 44.2% of the information while the 
# principal component 2 holds only 19% of the information. 
# Also, the other point to note is that while projecting 
# thirty-dimensional data to a two-dimensional data, 36.8% 
# information was lost.

# Let's plot the visualization of the 569 samples along the
# principal component - 1 and principal component - 2 axis. 
# It should give you good insight into how your samples are 
# distributed among the two classes.
from matplotlib import pyplot as plt

plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1']
               , principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})


# From the above graph, you can observe that the two classes
# benign and malignant, when projected to a two-dimensional 
# space, can be linearly separable up to some extent. 
# Other observations can be that the benign class is spread 
# out as compared to the malignant class.

# End of hands-on exercise!!!