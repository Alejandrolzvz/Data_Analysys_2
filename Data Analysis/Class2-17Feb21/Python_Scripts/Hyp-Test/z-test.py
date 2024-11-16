#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 16:33:17 2021

@author: franciscocantuortiz
"""

# Z-test
# one-sample Z-tezt

# https://towardsdatascience.com/hypothesis-testing-in-machine-learning-using-python-a0dc89e169ce 


from statsmodels.stats import weightstats as stests
import random
import numpy as np


df = random.sample(range(20, 50), 10)
df
df_mean = np.mean(df)
df_mean
# ztest ,pval = stests.ztest(df['bp_before'], x2=None, value=156)
ztest ,pval = stests.ztest(df, x2=None, value=30)
print(float(pval))
if pval<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")

# Two-sample Z test- In two sample z-test , similar to t-test here we are 
# checking two independent data groups and deciding whether sample mean of 
# two group is equal or not.
# H0 : mean of two group is 0
# H1 : mean of two group is not 0
# Example : we are checking in blood data after blood and before blood data.

df1 = random.sample(range(10, 30), 10)
df2 = random.sample(range(15, 40), 10)

# ztest ,pval1 = stests.ztest(df['bp_before'], x2=df['bp_after'], value=0,alternative='two-sided')
ztest ,pval1 = stests.ztest(df1, x2=df2, value=0,alternative='two-sided')

print(float(pval1))

if pval1<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")