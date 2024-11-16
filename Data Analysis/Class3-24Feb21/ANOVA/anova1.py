#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:20:17 2021

@author: franciscocantuortiz
"""

# https://datasciencechalktalk.com/2019/09/04/one-way-analysis-of-variance-anova-with-python/

# The t-test works well when dealing with two groups, but sometimes we want 
# to compare more than two groups at the same time. For example, if we wanted 
# to test whether voter age differs based on some categorical variable like 
# race, we have to compare the means of each level or group the variable. 
# We could carry out a separate t-test for each pair of groups, but when you 
# conduct many tests you increase the chances of false positives. 
# The analysis of variance or ANOVA is a statistical inference test that 
# lets you compare multiple groups at the same time.

# F = Between group variability / Within group variability

# As case study, imagine a bunch of students from different colleges taking 
# the same exam. You want to see if one college outperforms the other, 
# hence your null hypothesis is that the means of GPAs in each group are 
# equivalent to those of the other groups. To keep it simple, we will 
# consider 3 groups (college ‘A’, ‘B’, ‘C’) with 6 students each.

import pandas as pd

a=[25,25,27,30,23,20]
b=[30,30,21,24,26,28]
c=[18,30,29,29,24,26]
list_of_tuples = list(zip(a, b,c))
df = pd.DataFrame(list_of_tuples, columns = ['A', 'B', 'C'])
df
# We can look at this table as a matrix where the i-index is referring to 
# the students of same college, while the j-index is referring to the 
# group/college. Hence, the Yij entry will be referring to the ith student 
# of the jth college.

# Once made the following necessary assumptions:

# Response variable residuals are normally distributed (or approximately 
# normally distributed)
# Variances of populations are equal
# Responses for a given group are independent and identically distributed 
# normal random variables
# We can start with our steps:

import numpy as np
m1=np.mean(a)
m2=np.mean(b)
m3=np.mean(c)

print('Average mark for college A: {}'.format(m1))
print('Average mark for college B: {}'.format(m2))
print('Average mark for college C: {}'.format(m3))

m=(m1+m2+m3)/3
print('Overall mean: {}'.format(m))

# Compute the ‘between-group’ sum of squared differences (where n is 
# the number of observations per group/college, hence in our case n=6):

SSb=6*((m1-m)**2+(m2-m)**2+(m3-m)**2)
print('Between-groups Sum of Squared Differences: {}'.format(SSb))

# With those results, we can already compute one of the components of our 
# F-score, which is the between-group mean square value (MSb). Indeed, 
# knowing that the between-group degrees of freedom is k-1 (that means, 
# one less than the number of groups), we can compute the MSb as:

MSb=SSb/2
print('Between-groups Mean Square value: {}'.format(MSb))

# Calculate the “within-group” sum of squares.

err_a=list(a-m1)
err_b=list(b-m2)
err_c=list(c-m3)
err=err_a+err_b+err_c
ssw=[]
for i in err:
    ssw.append(i**2)
SSw=np.sum(ssw)

print('Within-group Sum of Squared Differences: {}'.format(SSw))

# Again, knowing that there are k(n-1) within-group degrees of freedom 
# (hence in our case 15), we can compute the within-group mean square value:

MSw=SSw/15
print('Within-group Mean Square value: {}'.format(MSw))

# We can finally compute the F-score, given by:

F=MSb/MSw
print('F-score: {}'.format(F))

# Let’s double-check this value with scipy:

import scipy.stats as stats

# Apply One-way ANOVA to A, B, C
stats.f_oneway(a,b,c)
# Nice, the two results coincide. If the assumptions above are true, 
# the ration MSb/MSw behaves as a Fisher distribution with (2,15) degrees 
# of freedom.


from scipy.stats import f
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
dfn, dfd = 2,15
x = np.linspace(f.ppf(0.01, dfn, dfd),f.ppf(0.99, dfn, dfd), 100)
ax.plot(x, f.pdf(x, dfn, dfd),'r-', lw=5, alpha=0.6, label='f pdf')


ax.plot(x, f.pdf(x, dfn, dfd),'r-', lw=5, alpha=0.6, label='f pdf')

# Let’s say we set alpha, which the level of significance, equal to 5%. 
# The corresponding F-critical value is 3.68. Hence:

from scipy.stats import f
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
dfn, dfd = 2,15
x = np.linspace(f.ppf(0.01, dfn, dfd),f.ppf(0.99, dfn, dfd), 100)
ax.plot(x, f.pdf(x, dfn, dfd),'r-', lw=5, alpha=0.6, label='f pdf')
plt.axvline(x=3.68, label='Critical value for alpha=0.05', color='g')
plt.axvline(x=F, label='F-score')
plt.legend()

# We do not reject the Null hypothesis about equality among means. 
# We can conclude (with an error of 5%, or alternatively, with a confidence 
# of 95%) that there is no significance difference between our three 
# colleges A, B and C.

