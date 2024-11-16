setwd("/Users/franciscocantuortiz/R")

install.packages("TTR")
library("TTR")

install.packages("forecast")
library("forecast")

# Example 1: Age of Death of Successive Kings of England
kings <- scan("http://robjhyndman.com/tsdldata/misc/kings.dat",skip=3)
kings
kingstimeseries <- ts(kings)
kingstimeseries
plot.ts(kingstimeseries)


# ARIMA models are defined for stationary time series. Therefore, if you start off 
# with a non-stationary time series, you will first need to ‘difference’ the time 
# series until you obtain a stationary time series. 
# If you have to difference the time series d times to obtain a stationary series, 
# then you have an ARIMA(p,d,q) model, where d is the order of differencing used.
kingtimeseriesdiff1 <- diff(kingstimeseries, differences=1)
plot.ts(kingtimeseriesdiff1)
acf(kingtimeseriesdiff1, lag.max=20)
acf(kingtimeseriesdiff1, lag.max=20, plot=FALSE)
pacf(kingtimeseriesdiff1, lag.max=20)
pacf(kingtimeseriesdiff1, lag.max=20, plot=FALSE)
auto.arima(kings)
kingstimeseriesarima <- arima(kingstimeseries, order=c(0,1,1))
kingstimeseriesarima
kingstimeseriesforecasts <- forecast(kingstimeseriesarima, h=5)
kingstimeseriesforecasts
plot(kingstimeseriesforecasts)
# acf(kingstimeseriesforecasts$residuals, lag.max=20)
acf(kingstimeseriesforecasts, lag.max = NULL, type = c("correlation", "covariance", "partial"), plot = TRUE, na.action = na.pass, demean = TRUE)
Box.test(kingstimeseriesforecasts$residuals, lag=20, type="Ljung-Box")
plot.ts(kingstimeseriesforecasts$residuals)
# Forecast function
plotForecastErrors <- function(forecasterrors) { 
  # make a histogram of the forecast errors: 
  mybinsize <- IQR(forecasterrors)/4
  mysd <- sd(forecasterrors)
  mymin <- min(forecasterrors) - mysd*5
  mymax <- max(forecasterrors) + mysd*3
  # generate normally distributed data with mean 0 and standard deviation 
  mynorm <- rnorm(10000, mean=0, sd=mysd) 
  mymin2 <- min(mynorm)
  mymax2 <- max(mynorm)
  if (mymin2 < mymin) { mymin <- mymin2 } 
  if (mymax2 > mymax) { mymax <- mymax2 } 
  # make a red histogram of the forecast errors, with the normally distributed ˓→data overlaid: 
  mybins <- seq(mymin, mymax, mybinsize)
  hist(forecasterrors, col="red", freq=FALSE, breaks=mybins)
  # freq=FALSE ensures the area under the histogram = 1
  # generate normally distributed data with mean 0 and standard deviation mysd 
  myhist <- hist(mynorm, plot=FALSE, breaks=mybins)
  # plot the normal curve as a blue line on top of the histogram of forecast errors:
  points(myhist$mids, myhist$density, type="l", col="blue", lwd=2) 
}
plotForecastErrors(kingstimeseriesforecasts$residuals)
# Since successive forecast errors do not seem to be correlated, and the forecast 
# errors seem to be normally distributed with mean zero and constant variance, 
# the ARIMA(0,1,1) does seem to provide an adequate predictive model for the ages 
# at death of English kings.

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

# Births Arima
birthstimeseriesseasonallyadjusteddiff1 <- diff(birthstimeseriesseasonallyadjusted, differences=1)
plot.ts(birthstimeseriesseasonallyadjusteddiff1)
acf(birthstimeseriesseasonallyadjusteddiff1, lag.max=20)
acf(birthstimeseriesseasonallyadjusteddiff1, lag.max=20, plot=FALSE)
pacf(birthstimeseriesseasonallyadjusteddiff1, lag.max=20)
pacf(birthstimeseriesseasonallyadjusteddiff1, lag.max=20, plot=FALSE)
auto.arima(birthstimeseriesseasonallyadjusted)
birthstimeseriesseasonallyadjustedarima <- arima(birthstimeseriesseasonallyadjusted, order=c(0,1,1))
birthstimeseriesseasonallyadjustedarima
birthstimeseriesseasonallyadjustedarima <- forecast(birthstimeseriesseasonallyadjustedarima, h=5)
birthstimeseriesseasonallyadjustedarima
plot(birthstimeseriesseasonallyadjustedarima)
acf(birthstimeseriesseasonallyadjustedarima$residuals, lag.max=20)
Box.test(birthstimeseriesseasonallyadjustedarima$residuals, lag=20, type="Ljung-Box")
plot.ts(birthstimeseriesseasonallyadjustedarima$residuals)
# Forecast function
plotForecastErrors <- function(forecasterrors) { 
  # make a histogram of the forecast errors: 
  mybinsize <- IQR(forecasterrors)/4
  mysd <- sd(forecasterrors)
  mymin <- min(forecasterrors) - mysd*5
  mymax <- max(forecasterrors) + mysd*3
  # generate normally distributed data with mean 0 and standard deviation 
  mynorm <- rnorm(10000, mean=0, sd=mysd) 
  mymin2 <- min(mynorm)
  mymax2 <- max(mynorm)
  if (mymin2 < mymin) { mymin <- mymin2 } 
  if (mymax2 > mymax) { mymax <- mymax2 } 
  # make a red histogram of the forecast errors, with the normally distributed ˓→data overlaid: 
  mybins <- seq(mymin, mymax, mybinsize)
  hist(forecasterrors, col="red", freq=FALSE, breaks=mybins)
  # freq=FALSE ensures the area under the histogram = 1
  # generate normally distributed data with mean 0 and standard deviation mysd 
  myhist <- hist(mynorm, plot=FALSE, breaks=mybins)
  # plot the normal curve as a blue line on top of the histogram of forecast errors:
  points(myhist$mids, myhist$density, type="l", col="blue", lwd=2) 
}
plotForecastErrors(birthstimeseriesseasonallyadjustedarima$residuals)
# Since successive forecast errors do not seem to be correlated, and the forecast 
# errors seem to be normally distributed with mean zero and constant variance, 
# the ARIMA(0,1,1) does seem to provide an adequate predictive model for the ages 
# at death of English kings.


# Example 4: Rain
# Forecasts using Simple Exponential Smoothing 
rain <- scan("http://robjhyndman.com/tsdldata/hurst/precip1.dat",skip=1)
rainseries <- ts(rain,start=c(1813))
plot.ts(rainseries)


# Example 5: Skirts
# Forecasting with Holt’s Exponential Smoothing
skirts <- scan("http://robjhyndman.com/tsdldata/roberts/skirts.dat",skip=5) 
skirtsseries <- ts(skirts,start=c(1866))
plot.ts(skirtsseries)


# Skirts Differencing Time Series
skirtsseriesdiff1 <- diff(skirtsseries, differences=1)
plot.ts(skirtsseriesdiff1)
skirtsseriesdiff2 <- diff(skirtsseries, differences=2)
plot.ts(skirtsseriesdiff2)

# Skirts Arima
acf(skirtsseriesdiff2, lag.max=20, plot=FALSE)
pacf(skirtsseriesdiff2, lag.max=20, plot=FALSE)
auto.arima(skirtsseriesdiff2)
skirtsseriesdiff2arima <- arima(skirtsseriesdiff2, order=c(0,1,1))
skirtsseriesdiff2arima
skirtsseriesdiff2arimaforecasts <- forecast(skirtsseriesdiff2arima, h=5)
skirtsseriesdiff2arimaforecasts
plot(skirtsseriesdiff2arimaforecasts)
acf(skirtsseriesdiff2arimaforecasts$residuals, lag.max=20)
Box.test(skirtsseriesdiff2arimaforecasts$residuals, lag=20, type="Ljung-Box")
plot.ts(skirtsseriesdiff2arimaforecasts$residuals)
plotForecastErrors <- function(forecasterrors) { 
  # make a histogram of the forecast errors: 
  mybinsize <- IQR(forecasterrors)/4
  mysd <- sd(forecasterrors)
  mymin <- min(forecasterrors) - mysd*5
  mymax <- max(forecasterrors) + mysd*3
  # generate normally distributed data with mean 0 and standard deviation 
  mynorm <- rnorm(10000, mean=0, sd=mysd) 
  mymin2 <- min(mynorm)
  mymax2 <- max(mynorm)
  if (mymin2 < mymin) { mymin <- mymin2 } 
  if (mymax2 > mymax) { mymax <- mymax2 } 
  # make a red histogram of the forecast errors, with the normally distributed ˓→data overlaid: 
  mybins <- seq(mymin, mymax, mybinsize)
  hist(forecasterrors, col="red", freq=FALSE, breaks=mybins)
  # freq=FALSE ensures the area under the histogram = 1
  # generate normally distributed data with mean 0 and standard deviation mysd 
  myhist <- hist(mynorm, plot=FALSE, breaks=mybins)
  # plot the normal curve as a blue line on top of the histogram of forecast errors:
  points(myhist$mids, myhist$density, type="l", col="blue", lwd=2) 
}
plotForecastErrors(skirtsseriesdiff2arimaforecasts$residuals)
