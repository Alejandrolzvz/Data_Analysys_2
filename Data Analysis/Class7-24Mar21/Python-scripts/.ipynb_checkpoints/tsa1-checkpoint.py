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

# The Data
# We are using Superstore sales data that can be downloaded from here.

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


# There are several categories in the Superstore sales data, we start from 
# time series analysis and forecasting for furniture sales.
df = pd.read_excel("Superstore.xls")
furniture = df.loc[df['Category'] == 'Furniture']

# We have a good 4-year furniture sales data.
furniture['Order Date'].min(), furniture['Order Date'].max()

# Data Preprocessing
# This step includes removing columns we do not need, check missing values, 
# aggregate sales by date and so on.
cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols, axis=1, inplace=True)
furniture = furniture.sort_values('Order Date')
furniture.isnull().sum()

furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()

# Indexing with Time Series Data
furniture = furniture.set_index('Order Date')
furniture.index

# Our current datetime data can be tricky to work with, therefore, we will 
# use the averages daily sales value for that month instead, and we are 
# using the start of each month as the timestamp.
y = furniture['Sales'].resample('MS').mean()

# Have a quick peek 2017 furniture sales data.
y['2017':]

# Visualizing Furniture Sales Time Series Data
y.plot(figsize=(15, 6))
plt.show()

# Some distinguishable patterns appear when we plot the data. The time-series 
# has seasonality pattern, such as sales are always low at the beginning of 
# the year and high at the end of the year. There is always an upward trend 
# within any single year with a couple of low months in the mid of the year.
# We can also visualize our data using a method called time-series decomposition
# that allows us to decompose our time series into three distinct components: 
# trend, seasonality, and noise.
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

# The plot above clearly shows that the sales of furniture is unstable, 
# along with its obvious seasonality.

# ----------------------------------------------------------------------------
# Time series forecasting with ARIMA
# We are going to apply one of the most commonly used method for time-series 
# forecasting, known as ARIMA, which stands for 
# Autoregressive Integrated Moving Average.
# ARIMA models are denoted with the notation ARIMA(p, d, q). 
# These three parameters account for seasonality, trend, and noise in data:
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

# This step is parameter Selection for our furniture’s sales 
# ARIMA Time Series Model. Our goal here is to use a “grid search” 
# to find the optimal set of parameters that yields the best 
# performance for our model.
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
                            continue  
                        
# The above output suggests that SARIMAX(1, 1, 1)x(1, 1, 0, 12) yields the 
# lowest AIC value of 297.78. Therefore we should consider this 
# to be optimal option.
# Fitting the ARIMA model
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

# We should always run model diagnostics to investigate any unusual behavior.

# It is not perfect, however, our model diagnostics suggests that the model 
# residuals are near normally distributed.

# Validating forecasts
# To help us understand the accuracy of our forecasts, we compare 
# predicted sales to real sales of the time series, and we set forecasts 
# to start at 2017–01–01 to the end of the data.
pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()

# The line plot is showing the observed values compared to the rolling 
# forecast predictions. Overall, our forecasts align with the true values 
# very well, showing an upward trend starts from the beginning of the 
# year and captured the seasonality toward the end of the year.
y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

# The Mean Squared Error of our forecasts is 22993.58
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

# The Root Mean Squared Error of our forecasts is 151.64

# In statistics, the mean squared error (MSE) of an estimator measures the 
# average of the squares of the errors — that is, the average squared 
# difference between the estimated values and what is estimated. 
# The MSE is a measure of the quality of an estimator — it is always 
# non-negative, and the smaller the MSE, the closer we are to finding the 
# line of best fit.
# Root Mean Square Error (RMSE) tells us that our model was able to forecast 
# the average daily furniture sales in the test set within 151.64 of 
# the real sales. Our furniture daily sales range from around 400 to over 1200.
#  In my opinion, this is a pretty good model so far.

# Producing and visualizing forecasts
pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()

# Our model clearly captured furniture sales seasonality. As we forecast 
# further out into the future, it is natural for us to become less confident 
# in our values. This is reflected by the confidence intervals generated 
# by our model, which grow larger as we move further out into the future.
# The above time series analysis for furniture makes me curious about 
# other categories, and how do they compare with each other over time. 
# Therefore, we are going to compare time series 
# of furniture and office supplier.

# --------------------------------------------------------------------------
# Time Series of Furniture vs. Office Supplies
# According to our data, there were way more number of sales from Office 
# Supplies than from Furniture over the years.
furniture = df.loc[df['Category'] == 'Furniture']
office = df.loc[df['Category'] == 'Office Supplies']
furniture.shape, office.shape

# Data Exploration
# We are going to compare two categories’ sales in the same time period. 
# This means combine two data frames into one and plot these 
# two categories’ time series into one plot.
cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols, axis=1, inplace=True)
office.drop(cols, axis=1, inplace=True)
furniture = furniture.sort_values('Order Date')
office = office.sort_values('Order Date')
furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()
office = office.groupby('Order Date')['Sales'].sum().reset_index()
furniture = furniture.set_index('Order Date')
office = office.set_index('Order Date')
y_furniture = furniture['Sales'].resample('MS').mean()
y_office = office['Sales'].resample('MS').mean()
furniture = pd.DataFrame({'Order Date':y_furniture.index, 'Sales':y_furniture.values})
office = pd.DataFrame({'Order Date': y_office.index, 'Sales': y_office.values})
store = furniture.merge(office, how='inner', on='Order Date')
store.rename(columns={'Sales_x': 'furniture_sales', 'Sales_y': 'office_sales'}, inplace=True)
store.head()

# plt.figure(figsize=(20, 8))
plt.plot(store['Order Date'], store['furniture_sales'], 'b-', label = 'furniture')
plt.plot(store['Order Date'], store['office_sales'], 'r-', label = 'office supplies')
plt.xlabel('Date'); plt.ylabel('Sales'); plt.title('Sales of Furniture and Office Supplies')
plt.legend();

# We observe that sales of furniture and office supplies shared a similar 
# seasonal pattern. Early of the year is the off season for both of the 
# two categories. It seems summer time is quiet for office supplies too. 
# in addition, average daily sales for furniture are higher than those of 
# office supplies in most of the months. It is understandable, as the value 
# of furniture should be much higher than those of office supplies. 
# Occasionally, office supplies passed furniture on average daily sales. 
# Let’s find out when was the first time office supplies’ sales surpassed 
# those of furniture’s.
# first_date = store.ix[np.min(list(np.where(store['office_sales'] > store['furniture_sales'])[0])), 'Order Date']
# print("Office supplies first time produced higher sales than furniture is {}.".format(first_date.date()))

# Office supplies first time produced higher sales than furniture is 2014–07–01.
# It was July 2014!


