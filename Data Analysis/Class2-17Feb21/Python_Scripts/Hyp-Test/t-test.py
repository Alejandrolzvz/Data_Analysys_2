#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:35:50 2021

@author: franciscocantuortiz
"""


# One sample t-test : The One Sample t Test determines whether the sample 
# mean is statistically different from a known or hypothesised population 
# mean. The One Sample t Test is a parametric test.
# Example :- you have 30 ages and you are checking whether avg age is 25 or not
from scipy.stats import ttest_1samp
import numpy as np
import random
# ages = np.genfromtxt(“ages.csv”)
ages = random.sample(range(20, 50), 30)
print(ages)
ages_mean = np.mean(ages)
print(ages_mean)
tset, pval = ttest_1samp(ages, 25)
print('p-values',pval)
if pval < 0.05:    # alpha value is 0.05 or 5%
   print(" we are rejecting null hypothesis")
else:
  print("we are accepting null hypothesis")
  
# Two-sambpel t-test
  
from scipy.stats import ttest_ind
import numpy as np
import random

#Generate 10 random numbers between 10 and 30
week1 = random.sample(range(10, 30), 10)
print("week1 data :-\n")
print(week1)

#Generate 10 random numbers between 15 and 35
week2 = random.sample(range(10, 30), 10)
print("week2 data :-\n")
print(week2)

#week1 = np.genfromtxt("week1.csv",  delimiter=",")
#week2 = np.genfromtxt("week2.csv",  delimiter=",")

week1_mean = np.mean(week1)
week2_mean = np.mean(week2)
print("week1 mean value:",week1_mean)
print("week2 mean value:",week2_mean)
week1_std = np.std(week1)
week2_std = np.std(week2)
print("week1 std value:",week1_std)
print("week2 std value:",week2_std)
ttest,pval = ttest_ind(week1,week2)
print("p-value",pval)
if pval <0.05:
  print("we reject null hypothesis")
else:
  print("we accept null hypothesis")
  
# Paired sampled t-test
# The paired sample t-test is also called dependent sample t-test. 
# It’s an uni variate test that tests for a significant difference between 
# 2 related variables. An example of this is if you where to collect the 
# blood pressure for an individual before and after some treatment, 
# condition, or time point.
# H0 :- means difference between two sample is 0
# H1:- mean difference between two sample is not 0
  
import pandas as pd
from scipy import stats
# df = pd.read_csv("blood_pressure.csv")
# df[['bp_before','bp_after']].describe()
# ttest,pval = stats.ttest_rel(df['bp_before'], df['bp_after'])
df1 = random.sample(range(10, 30), 10)
df2 = random.sample(range(15, 40), 10)
ttest,pval = stats.ttest_rel(df1, df2)
print(pval)
if pval<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")