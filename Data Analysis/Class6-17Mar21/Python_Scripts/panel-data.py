#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 19:16:57 2021

@author: franciscocantuortiz
"""

# https://bashtage.github.io/linearmodels/panel/examples/examples.html


# These examples all make use of the wage panel from

# Vella and M. Verbeek (1998), “Whose Wages Do Unions Raise? A Dynamic Model 
# of Unionism and Wage Rate Determination for Young Men,” Journal of 
# Applied Econometrics 13, 163-183.

# The data set consists of wages and characteristics for men during the 1980s.
# The entity identifier is nr and the time identified is year. 

# Here a MultiIndex DataFrame is used to hold the data in a format that can 
# be understood as a panel. Before setting the index, a year Categorical 
# is created which facilitated making dummies.


pip install linearmodels
from linearmodels.datasets import wage_panel
import pandas as pd

data = wage_panel.load()
year = pd.Categorical(data.year)
data = data.set_index(["nr", "year"])
data["year"] = year
print(wage_panel.DESCR)
print(data.head())

# Basic regression on panel data¶
# PooledOLS is just plain OLS that understands that various panel data 
# structures. It is useful as a base model. Here the log wage is modeled 
# using all of the variables and time dummies.

from linearmodels.panel import PooledOLS
import statsmodels.api as sm

exog_vars = ["black", "hisp", "exper", "expersq", "married", "educ", "union", "year"]
exog = sm.add_constant(data[exog_vars])
mod = PooledOLS(data.lwage, exog)
pooled_res = mod.fit()
print(pooled_res)

# Estimating parameters with uncorrelated effects¶
# When modeling panel data it is common to consider models beyond what 
# OLS will efficiently estimate. The most common are error component models 
# which add an additional term to the standard OLS model,

# 𝑦𝑖𝑡=𝑥𝑖𝑡𝛽+𝛼𝑖+𝜖𝑖𝑡
# where 𝛼𝑖 affects all values of entity i. When the 𝛼𝑖 are uncorrelated 
# with the regressors in 𝑥𝑖𝑡, a random effects model can be used to 
# efficiently estimate parameters of this model.

# Random effects¶
# The random effects model is virtually identical to the pooled OLS model 
# except that is accounts for the structure of the model and so is more 
# efficient. Random effects uses a quasi-demeaning strategy which subtracts 
# the time average of the within entity values to account for the common shock.

from linearmodels.panel import RandomEffects
mod = RandomEffects(data.lwage, exog)
re_res = mod.fit()
print(re_res)

# The model fit is fairly similar, although the return to experience has 
# changed substantially, as has its significance. This is partially 
# explainable by the inclusion of the year dummies which will fit the trend 
# in experience and so only the cross-sectional differences matter. 
# The quasi-differencing in the random effects estimator depends on 
# a quantity that depends on the relative variance of the idiosyncratic 
# shock and the common shock. This can be accessed using variance_decomposition.


# Variance Decomposition

re_res.variance_decomposition

# Demeaning Theta
re_res.theta.head()


# Between Estimation

from linearmodels.panel import BetweenOLS

exog_vars = ["black", "hisp", "exper", "married", "educ", "union"]
exog = sm.add_constant(data[exog_vars])
mod = BetweenOLS(data.lwage, exog)
be_res = mod.fit()
print(be_res)

# Fixed Effects
from linearmodels.panel import PanelOLS

exog_vars = ["expersq", "union", "married", "year"]
exog = sm.add_constant(data[exog_vars])
mod = PanelOLS(data.lwage, exog, entity_effects=True)
fe_res = mod.fit()
print(fe_res)

# Time Effects
# Effects VS Dummies
from linearmodels.panel import PanelOLS

exog_vars = ["expersq", "union", "married"]
exog = sm.add_constant(data[exog_vars])
mod = PanelOLS(data.lwage, exog, entity_effects=True, time_effects=True)
fe_te_res = mod.fit()
print(fe_te_res)


# First Difference OLS
from linearmodels.panel import FirstDifferenceOLS

exog_vars = ["exper", "expersq", "union", "married"]
exog = data[exog_vars]
mod = FirstDifferenceOLS(data.lwage, exog)
fd_res = mod.fit()
print(fd_res)

# Comparing Panel Data Models

from linearmodels.panel import compare

print(compare({"BE": be_res, "RE": re_res, "Pooled": pooled_res}))

