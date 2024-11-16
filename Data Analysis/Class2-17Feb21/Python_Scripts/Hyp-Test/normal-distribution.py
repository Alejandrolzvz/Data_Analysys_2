#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:14:51 2021

@author: franciscocantuortiz
"""

# Normal Distribution
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html


import numpy as np
mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)

# Verify values
abs(mu - np.mean(s))
abs(sigma - np.std(s, ddof=1))

# Plot the distribution
import matplotlib.pyplot as plt

count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()

# np.random.normal(3, 2.5, size=(2, 4))
