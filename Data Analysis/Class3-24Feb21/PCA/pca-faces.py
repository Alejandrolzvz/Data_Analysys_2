#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:10:13 2021

@author: franciscocantuortiz
"""

# Face recognition using PCA

# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html 

# Loading faces file takes about 4 hours!!!

from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)

# Display dataset information
print(faces.target_names)
print(faces.images.shape)

# Let's take a look at the principal axes that span this dataset. Because 
# this is a large dataset, we will use RandomizedPCA—it contains a 
# randomized method to approximate the first $N$ principal components much 
# more quickly than the standard PCA estimator, and thus is very useful for 
# high-dimensional data (here, a dimensionality of nearly 3,000). We will 
# take a look at the first 150 components:

import sklearn
import sklearn.decomposition
from sklearn.decomposition import PCA
# import RandomizedPCA

# from sklearn.decomposition import RandomizedPCA
pca = PCA(150)
pca.fit(faces.data)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone')

# In this case, it can be interesting to visualize the images associated with
# the first several principal components (these components are technically 
# known as "eigenvectors," so these types of images are often called 
# "eigenfaces"). As you can see in this figure, they are as creepy as they 
# sound.

# 
    
    
# We apply now PCA to extract the more informative attributes

# Compute the components and projected faces
pca = PCA(150).fit(faces.data)
components = pca.transform(faces.data)
projected = pca.inverse_transform(components)

# Plot the results
fig, ax = plt.subplots(2, 10, figsize=(10, 2.5),
                       subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(10):
    ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap='binary_r')
    ax[1, i].imshow(projected[i].reshape(62, 47), cmap='binary_r')
    
ax[0, 0].set_ylabel('full-dim\ninput')
ax[1, 0].set_ylabel('150-dim\nreconstruction');

# The results are very interesting, and give us insight into how the images 
# vary: for example, the first few eigenfaces (from the top left) seem to be 
# associated with the angle of lighting on the face, and later principal 
# vectors seem to be picking out certain features, such as eyes, noses, and 
# lips. Let's take a look at the cumulative variance of these components to 
# see how much of the data information the projection is preserving:

import numpy as np

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

# We see that these 150 components account for just over 90% of the variance. 
# That would lead us to believe that using these 150 components, we would 
# recover most of the essential characteristics of the data. To make this 
# more concrete, we can compare the input images with the images reconstructed
# from these 150 components:


