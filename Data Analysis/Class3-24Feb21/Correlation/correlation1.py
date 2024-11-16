#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 09:26:25 2021

@author: franciscocantuortiz
"""

# generate related variables
from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot
# seed random number generator
seed(1)
# prepare data
data1 = 20 * randn(1000) + 100
data2 = data1 + (10 * randn(1000) + 50)
# summarize
print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
# plot
pyplot.scatter(data1, data2)
pyplot.show()

# calculate the covariance between two variables
from numpy import cov

covariance = cov(data1, data2)
print(covariance)


# calculate the Pearson's correlation between two variables
from scipy.stats import pearsonr

corr, p = pearsonr(data1, data2)
print('Pearsons correlation: %.3f' % corr)
print('p-value: %.3f' % p)

# interpret the significance
alpha = 0.05
if p > alpha:
	print('Samples are Pearson uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('Samples are Pearson correlated (reject H0) p=%.3f' % p)
    
    
# calculate the spearmans's correlation between two variables
from scipy.stats import spearmanr

corr, p = spearmanr(data1, data2)
print('Spearmans correlation: %.3f' % corr)
print('p-value: %.3f' % p)

# interpret the significance
alpha = 0.05
if p > alpha:
	print('Samples are Spearman uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('Samples are Spearman correlated (reject H0) p=%.3f' % p)

    
# Kendall Correlation
# calculate the kendall's correlation between two variables
from scipy.stats import kendalltau

coef, p = kendalltau(data1, data2)
print('Kendall correlation coefficient: %.3f' % coef)
print('p-value: %.3f' % p)

# interpret the significance
alpha = 0.05
if p > alpha:
	print('Samples are Kendall uncorrelated (fail to reject H0) p=%.3f' % p)
else:
	print('Samples are Kendall correlated (reject H0) p=%.3f' % p)