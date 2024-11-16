#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:15:34 2021

@author: franciscocantuortiz
"""

# PCA for visualization: Hand-written digits

# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html 

# Load tyhe data
from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

# Recall that the data consists of 8×8 pixel images, meaning that 
# they are 64-dimensional. To gain some intuition into the 
# relationships between these points, we can use PCA to project 
# them to a more manageable number of dimensions, say two:

from sklearn.decomposition import PCA
pca = PCA(2)  # project from 64 to 2 dimensions
projected = pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)

# We can now plot the first two principal components of each point 
# to learn about the data:
# Visualize 2D projection
import numpy as np
import matplotlib.pyplot as plt

plt.scatter(projected[:, 0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();

# import pandas as pd

# Define plotting function
def plot_pca_components(x, coefficients=None, mean=0, components=None,
                        imshape=(8, 8), n_components=8, fontsize=12,
                        show_mean=True):
    if coefficients is None:
        coefficients = x
        
    if components is None:
        components = np.eye(len(coefficients), len(x))
        
    mean = np.zeros_like(x) + mean
        

    fig = plt.figure(figsize=(1.2 * (5 + n_components), 1.2 * 2))
    g = plt.GridSpec(2, 4 + bool(show_mean) + n_components, hspace=0.3)

    def show(i, j, x, title=None):
        ax = fig.add_subplot(g[i, j], xticks=[], yticks=[])
        ax.imshow(x.reshape(imshape), interpolation='nearest')
        if title:
            ax.set_title(title, fontsize=fontsize)

    show(slice(2), slice(2), x, "True")
    
    approx = mean.copy()
    
    counter = 2
    if show_mean:
        show(0, 2, np.zeros_like(x) + mean, r'$\mu$')
        show(1, 2, approx, r'$1 \cdot \mu$')
        counter += 1

    for i in range(n_components):
        approx = approx + coefficients[i] * components[i]
        show(0, i + counter, components[i], r'$c_{0}$'.format(i + 1))
        show(1, i + counter, approx,
             r"${0:.2f} \cdot c_{1}$".format(coefficients[i], i + 1))
        if show_mean or i > 0:
            plt.gca().text(0, 1.05, '$+$', ha='right', va='bottom',
                           transform=plt.gca().transAxes, fontsize=fontsize)

    show(slice(2), slice(-2, None), approx, "Approx")
    return fig

# Plotting digits
from sklearn.datasets import load_digits
import seaborn as sns

digits = load_digits()
sns.set_style('white')

# To construct the image, we multiply each element of the vector by the pixel 
# it describes, and then add the results together to build the image:

# image(x)=x1⋅(pixel 1)+x2⋅(pixel 2)+x3⋅(pixel 3)⋯x64⋅(pixel 64)

# One way we might imagine reducing the dimension of this data is to zero out
# all but a few of these basis vectors. For example, if we use only the 
# first eight pixels, we get an eight-dimensional projection of the data, 
# but it is not very reflective of the whole image: we've thrown out nearly 
# 90% of the pixels!

fig = plot_pca_components(digits.data[10],show_mean=False)

# The upper row of panels shows the individual pixels, and the lower row 
# shows the cumulative contribution of these pixels to the construction of 
# the image. Using only eight of the pixel-basis components, we can only 
# construct a small portion of the 64-pixel image. Were we to continue this 
# sequence and use all 64 pixels, we would recover the original image.

# Insted, we take the mean + the 8 pixels that carry out the major variance 
# given by PCA

# PCA can be thought of as a process of choosing optimal basis functions, 
# such that adding together just the first few of them is enough to 
# suitably reconstruct the bulk of the elements in the dataset. 
# The principal components, which act as the low-dimensional representation 
# of our data, are simply the coefficients that multiply each of the 
# elements in this series. This figure shows a similar depiction of 
# reconstructing this digit using the mean plus the first eight PCA basis 
# functions:

# image(x)=mean+x1⋅(basis 1)+x2⋅(basis 2)+x3⋅(basis 3)⋯

pca = PCA(n_components=8)
Xproj = pca.fit_transform(digits.data)
sns.set_style('white')
fig = plot_pca_components(digits.data[10], Xproj[10],pca.mean_, pca.components_)

pca.explained_variance_ratio_
