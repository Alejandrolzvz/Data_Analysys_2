#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 23:47:09 2021

@author: franciscocantuortiz
"""

# https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/

# Example of the Shapiro-Wilk Normality Test
from scipy.stats import shapiro
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
stat, p = shapiro(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably Gaussian')
else:
	print('Probably not Gaussian')

# Example of the D'Agostino's K^2 Normality Test
from scipy.stats import normaltest
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
stat, p = normaltest(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably Gaussian')
else:
	print('Probably not Gaussian')
    

# Example of the Anderson-Darling Normality Test
from scipy.stats import anderson
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
result = anderson(data)
print('stat=%.3f' % (result.statistic))
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < cv:
		print('Probably Gaussian at the %.1f%% level' % (sl))
	else:
		print('Probably not Gaussian at the %.1f%% level' % (sl))
        
        

# t-Student Test

# Tests whether the means of two independent samples are significantly 
# different.

# Assumptions

# Observations in each sample are independent and identically distributed (iid).
# Observations in each sample are normally distributed.
# Observations in each sample have the same variance.
# Interpretation

# H0: the means of the samples are equal.
# H1: the means of the samples are unequal.
# Python Code

# Example of the Student's t-test
from scipy.stats import ttest_ind
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = ttest_ind(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')


# Paired Student’s t-test
# Tests whether the means of two paired samples are significantly different.

# Assumptions

# Observations in each sample are independent and identically distributed (iid).
# Observations in each sample are normally distributed.
# Observations in each sample have the same variance.
# Observations across each sample are paired.
# Interpretation

# H0: the means of the samples are equal.
# H1: the means of the samples are unequal.
# Python Code

# Example of the Paired Student's t-test
from scipy.stats import ttest_rel
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = ttest_rel(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')




# ANOVA
# Tests whether the means of two or more independent samples are 
# significantly different.

# Assumptions

# Observations in each sample are independent and identically distributed (iid).
# Observations in each sample are normally distributed.
# Observations in each sample have the same variance.
# Interpretation

# H0: the means of the samples are equal.
# H1: one or more of the means of the samples are unequal.
       
# Example of the Analysis of Variance Test
from scipy.stats import f_oneway
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
data3 = [-0.208, 0.696, 0.928, -1.148, -0.213, 0.229, 0.137, 0.269, -0.870, -1.204]
stat, p = f_oneway(data1, data2, data3)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

# Correlation Analysis
    
# Pearson’s Correlation Coefficient
# Tests whether two samples have a linear relationship.

# Assumptions

# Observations in each sample are independent and identically distributed (iid).
# Observations in each sample are normally distributed.
# Observations in each sample have the same variance.
# Interpretation

# H0: the two samples are independent.
# H1: there is a dependency between the samples.
# Python Code

# Example of the Pearson's Correlation test
from scipy.stats import pearsonr
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
stat, p = pearsonr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')



# Spearman’s Rank Correlation
# Tests whether two samples have a monotonic relationship.

# Assumptions

# Observations in each sample are independent and identically distributed (iid).
# Observations in each sample can be ranked.
# Interpretation

# H0: the two samples are independent.
# H1: there is a dependency between the samples.
# Python Code

# Example of the Spearman's Rank Correlation Test
from scipy.stats import spearmanr
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
stat, p = spearmanr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Kendall’s Rank Correlation
# Tests whether two samples have a monotonic relationship.

# Assumptions

# Observations in each sample are independent and identically distributed (iid).
# Observations in each sample can be ranked.
# Interpretation

# H0: the two samples are independent.
# H1: there is a dependency between the samples.
# Python Code

# Example of the Kendall's Rank Correlation Test
from scipy.stats import kendalltau
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
stat, p = kendalltau(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')

# Chi-Squared Test
# Tests whether two categorical variables are related or independent.

# Assumptions

# Observations used in the calculation of the contingency table are independent.
# 25 or more examples in each cell of the contingency table.
# Interpretation

# H0: the two samples are independent.
# H1: there is a dependency between the samples.
# Python Code

# Example of the Chi-Squared Test
from scipy.stats import chi2_contingency
table = [[10, 20, 30],[6,  9,  17]]
stat, p, dof, expected = chi2_contingency(table)

print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
