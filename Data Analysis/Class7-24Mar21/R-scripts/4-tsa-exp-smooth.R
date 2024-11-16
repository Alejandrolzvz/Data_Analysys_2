setwd("/Users/franciscocantuortiz/R")

install.packages("TTR")
library("TTR")

install.packages("forecast")
library("forecast")

# Example 1: Age of Death of Successive Kings of England
# getwd()
kings <- scan("http://robjhyndman.com/tsdldata/misc/kings.dat",skip=3)

kingstimeseries <- ts(kings, frequency = 12)
kingstimeseries
plot.ts(kingstimeseries)

# Decomposing the Kings time series
kingstimeseriescomponents <- decompose(kingstimeseries)
kingstimeseriescomponents$seasonal
plot(kingstimeseriescomponents)

# Kings Seasonally Adjusting
kingstimeseriesseasonallyadjusted <- kingstimeseries - kingstimeseriescomponents$seasonal
plot(kingstimeseriesseasonallyadjusted)

# The Kings Age of Death time series seems to be not seasonal 
# and be analyzed with an additive model
# To estimate the trend component of a non-seasonal time series that can be 
# described using an additive model, it is common to use a smoothing method, 
# such as calculating the simple moving average of the time series.

# Simple Moving Average of a TS

kingstimeseriesSMA3 <- SMA(kingstimeseries,n=3)
plot.ts(kingstimeseriesSMA3)
kingstimeseriesSMA8 <- SMA(kingstimeseries,n=8)
plot.ts(kingstimeseriesSMA8)


# If you have a time series that can be described using an additive model with 
# constant level and no seasonality, you can use simple exponential smoothing to make 
# short-term forecasts.
# The simple exponential smoothing method provides a way of estimating the level at 
# the current time point. Smoothing is controlled by the parameter alpha; for the 
# estimate of the level at the current time point. 
# The value of alpha; lies between 0 and 1. Values of alpha that are close to 0 mean 
# that little weight is placed on the most recent observations when making forecasts 
# of future values.

# Holt-Winter Exponential Smoothing

kingstimeseriesforecasts <- HoltWinters(kingstimeseries, beta=FALSE, gamma=FALSE)
kingstimeseriesforecasts
kingstimeseriesforecasts$fitted
plot(kingstimeseriesforecasts)
kingstimeseriesforecasts$SSE
kingstimeseriesforecasts2 <- forecast(kingstimeseriesforecasts, h=8)
kingstimeseriesforecasts2
plot(kingstimeseriesforecasts2)


# Example 2
# Births per month in New York city, from January 1946 to December 1959
births <- scan("http://robjhyndman.com/tsdldata/data/nybirths.dat")
birthstimeseries <- ts(births, frequency=12, start=c(1946,1))
birthstimeseries
plot(birthstimeseries)

# Decomposing the births TS
birthstimeseriescomponents <- decompose(birthstimeseries)
birthstimeseriescomponents$seasonal
plot(birthstimeseriescomponents)

# Births Seasonally Adjusting
birthstimeseriesseasonallyadjusted <- birthstimeseries - birthstimeseriescomponents$seasonal
plot(birthstimeseriesseasonallyadjusted)

# Births Forecasting with Holt-Winters' Exponential Smoothing

birthstimeseriesseasonallyadjustedforecasts <- HoltWinters(birthstimeseriesseasonallyadjusted, beta=FALSE, gamma=FALSE)
birthstimeseriesseasonallyadjustedforecasts
birthstimeseriesseasonallyadjustedforecasts$fitted
plot(birthstimeseriesseasonallyadjustedforecasts)
birthstimeseriesseasonallyadjustedforecasts$SSE
birthstimeseriesseasonallyadjustedforecasts2 <- forecast(birthstimeseriesseasonallyadjustedforecasts, h=8)
birthstimeseriesseasonallyadjustedforecasts2
plot(birthstimeseriesseasonallyadjustedforecasts2)


# Example 3
# Monthly sales for a souvenir shop at a beach resort town in Queensland, Australia, 
# for January 1987-December 1993
souvenir <- scan("http://robjhyndman.com/tsdldata/data/fancy.dat")
souvenirtimeseries <- ts(souvenir, frequency=12, start=c(1987,1))
souvenirtimeseries
plot.ts(souvenirtimeseries)
logsouvenirtimeseries <- log(souvenirtimeseries)
plot.ts(logsouvenirtimeseries)

# Souvenirs Forecasting with Holt-Winters' Exponential Smoothing

souvenirtimeseriesforecasts <- HoltWinters(logsouvenirtimeseries)
souvenirtimeseriesforecasts
plot(souvenirtimeseriesforecasts)
souvenirtimeseriesforecasts2 <- forecast(souvenirtimeseriesforecasts,h=48)
plot(souvenirtimeseriesforecasts2)
acf(souvenirtimeseriesforecasts2$residuals, lag.max = NULL, type = c("correlation", "covariance", "partial"), plot = TRUE, na.action = na.pass, demean = TRUE)
Box.test(souvenirtimeseriesforecasts2$residuals, lag=20, type="Ljung-Box")
plot.ts(souvenirtimeseriesforecasts2$residuals)


# Example 4: Rain
# Forecasts using Simple Exponential Smoothing 
rain <- scan("http://robjhyndman.com/tsdldata/hurst/precip1.dat",skip=1)
rainseries <- ts(rain,start=c(1813))
plot.ts(rainseries)

# Rain Forecasting with Holt-Winters' Exponential Smoothing
rainseriesforecasts <- HoltWinters(rainseries, beta=FALSE, gamma=FALSE) 
rainseriesforecasts
rainseriesforecasts$fitted 
plot(rainseriesforecasts)
rainseriesforecasts$SSE 
HoltWinters(rainseries, beta=FALSE, gamma=FALSE, l.start=23.56)

# Forecast Package
rainseriesforecasts2 <- forecast(rainseriesforecasts, h=8)
rainseriesforecasts2
plot(rainseriesforecasts2)
acf(rainseriesforecasts2$residuals, lag.max = NULL, type = c("correlation", "covariance", "partial"), plot = TRUE, na.action = na.pass, demean = TRUE)
Box.test(rainseriesforecasts2$residuals, lag=20, type="Ljung-Box")
plot.ts(rainseriesforecasts2$residuals)


# Example 5: Skirts
# Forecasting with Holt’s Exponential Smoothing
skirts <- scan("http://robjhyndman.com/tsdldata/roberts/skirts.dat",skip=5) 
skirtsseries <- ts(skirts,start=c(1866))
plot.ts(skirtsseries)
skirtsseriesforecasts <- HoltWinters(skirtsseries, gamma=FALSE)
skirtsseriesforecasts
skirtsseriesforecasts$SSE
plot(skirtsseriesforecasts)
HoltWinters(skirtsseries, gamma=FALSE, l.start=608, b.start=9)
skirtsseriesforecasts2 <- forecast(skirtsseriesforecasts, h=19)
plot(skirtsseriesforecasts2)
acf(skirtsseriesforecasts2$residuals, lag.max = NULL, type = c("correlation", "covariance", "partial"), plot = TRUE, na.action = na.pass, demean = TRUE)
Box.test(skirtsseriesforecasts2$residuals, lag=20, type="Ljung-Box")
plot.ts(skirtsseriesforecasts2$residuals)

