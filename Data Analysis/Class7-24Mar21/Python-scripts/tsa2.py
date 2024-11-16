#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:28:26 2021

@author: franciscocantuortiz
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:26:12 2021

@author: franciscocantuortiz
"""

# https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b


# Time series analysis comprises methods for analyzing time series data in 
# order to extract meaningful statistics and other characteristics of the data.
#  Time series forecasting is the use of a model to predict future values 
# based on previously observed values.
# Time series are widely used for non-stationary data, like economic, 
# weather, stock price, and retail sales in this post. We will demonstrate 
# different approaches for forecasting retail sales time series. 
# Let’s get started!


# -------------------------------------------------------------------------
# Time Series Modeling with Prophet
# Released by Facebook in 2017, forecasting tool Prophet is designed for 
# analyzing time-series that display patterns on different time scales such 
# as yearly, weekly and daily. It also has advanced capabilities for modeling 
# the effects of holidays on a time-series and implementing custom changepoints.
#  Therefore, we are using Prophet to get a model up and running.

# bash

conda install gcc 
conda install -c conda-forge fbprophet

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
from fbprophet import Prophet

# The Data
# We are using Superstore sales data that can be downloaded from here.

df = pd.read_excel("Superstore.xls")
furniture = df.loc[df['Category'] == 'Furniture']
office = df.loc[df['Category'] == 'Office Supplies']
furniture.shape, office.shape

furniture = furniture.rename(columns={'Order Date': 'ds', 'Sales': 'y'})
furniture_model = Prophet(interval_width=0.95)
furniture_model.fit(furniture)
office = office.rename(columns={'Order Date': 'ds', 'Sales': 'y'})
office_model = Prophet(interval_width=0.95)
office_model.fit(office)
furniture_forecast = furniture_model.make_future_dataframe(periods=36, freq='MS')
furniture_forecast = furniture_model.predict(furniture_forecast)
office_forecast = office_model.make_future_dataframe(periods=36, freq='MS')
office_forecast = office_model.predict(office_forecast)

plt.figure(figsize=(18, 6))
furniture_model.plot(furniture_forecast, xlabel = 'Date', ylabel = 'Sales')
plt.title('Furniture Sales');

plt.figure(figsize=(18, 6))
office_model.plot(office_forecast, xlabel = 'Date', ylabel = 'Sales')
plt.title('Office Supplies Sales');

# Compare Forecasts
# We already have the forecasts for three years for these two categories into 
# the future. We will now join them together to compare their future forecasts.
furniture_names = ['furniture_%s' % column for column in furniture_forecast.columns]
office_names = ['office_%s' % column for column in office_forecast.columns]
merge_furniture_forecast = furniture_forecast.copy()
merge_office_forecast = office_forecast.copy()
merge_furniture_forecast.columns = furniture_names
merge_office_forecast.columns = office_names
forecast = pd.merge(merge_furniture_forecast, merge_office_forecast, how = 'inner', left_on = 'furniture_ds', right_on = 'office_ds')
forecast = forecast.rename(columns={'furniture_ds': 'Date'}).drop('office_ds', axis=1)
forecast.head()

# Trend and Forecast Visualization
plt.figure(figsize=(10, 7))
plt.plot(forecast['Date'], forecast['furniture_trend'], 'b-')
plt.plot(forecast['Date'], forecast['office_trend'], 'r-')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Sales')
plt.title('Furniture vs. Office Supplies Sales Trend');

# plt.figure(figsize=(10, 7))
plt.plot(forecast['Date'], forecast['furniture_yhat'], 'b-')
plt.plot(forecast['Date'], forecast['office_yhat'], 'r-')
plt.legend(); plt.xlabel('Date'); plt.ylabel('Sales')
plt.title('Furniture vs. Office Supplies Estimate');

# Trends and Patterns
# Now, we can use the Prophet Models to inspect different trends of these 
# two categories in the data.
furniture_model.plot_components(furniture_forecast);

office_model.plot_components(office_forecast);

# Good to see that the sales for both furniture and office supplies have 
# been linearly increasing over time and will be keep growing, although 
# office supplies’ growth seems slightly stronger.
# The worst month for furniture is April, the worst month for 
# office supplies is February. The best month for furniture is December, 
# and the best month for office supplies is October.
# There are many time-series analysis we can explore from now on, such as 
# forecast with uncertainty bounds, change point and anomaly detection, 
# forecast time-series with external data source. We have only just started.
# Source code can be found on Github. I look forward to hearing 
# feedback or questions.
# References:
# A Guide to Time Series Forecasting with ARIMA in Python 3
# A Guide to Time Series Forecasting with Prophet in Python 3
# Susan Li
# Changing the world, one post at a time. 
# Sr Data Scientist, Toronto Canada. https://www.linkedin.com/in/susanli/
