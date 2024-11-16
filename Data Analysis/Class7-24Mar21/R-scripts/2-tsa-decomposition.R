setwd("/Users/franciscocantuortiz/R")

# Example 1: Age of Death of Successive Kings of England
# starting with William the Conqueror
# Source: McNeill, "Interactive Data Analysis"getwd()
kings <- scan("http://robjhyndman.com/tsdldata/misc/kings.dat",skip=3)
kings
kingstimeseries <- ts(kings)
kingstimeseries
plot(kingstimeseries)

# Decomposing the Kings time series
kingstimeseriescomponents <- decompose(kingstimeseries)

kingstimeseries <- ts(kings, frequency = 12)

# Decomposing the Kings time series
kingstimeseriescomponents <- decompose(kingstimeseries)
kingstimeseriescomponents$seasonal
plot(kingstimeseriescomponents)


# --------------------------------------------------------------

# Example 2: Covid-19 Confirmed casews
mydata <- read.csv("~/R/novel-corona-virus-2019-dataset/covid_19_data.csv")
mydata
confirmed<-c(444,	444,	549,	761,	1058,	1423,	3554,	3554,	4903,	5806,	7153,	11177,	13522,	16678,	19665,	22112,	24953,	27100,	29631,	31728,	33366,	33366,	48206,	54406,	56249,	58182,	59989,	61682,	62031,	62442,	62662,	64084,	64084,	64287,	64786,	65187,	65596,	65914,	66337,	66907,	67103,	67217,	67332,	67466,	67592,	67666,	67707,	67743,	67760,	67773,	67781,	67786,	67790,	67794,	67798,	67799,	67800,	67800,	67800,	67800,	67800,	67800,	67801,	67801,	67801,	67801,	67801,	67801,	67801)
confirmed
confirmedtimeseries<-ts(confirmed, frequency=4)
confirmedtimeseries
plot(confirmedtimeseries)

# Decomposing the Confirmed time series
confirmedtimeseriescomponents <- decompose(confirmedtimeseries)
confirmedtimeseriescomponents$seasonal
plot(confirmedtimeseriescomponents)

# --------------------------------------------------------------

# Example 3: Births

# Births per month in New York city, from January 1946 to December 1959
births <- scan("http://robjhyndman.com/tsdldata/data/nybirths.dat")
birthstimeseries <- ts(births, frequency=12, start=c(1946,1))
birthstimeseries
plot(birthstimeseries)

# Decomposing the births time series
birthstimeseriescomponents <- decompose(birthstimeseries)
birthstimeseriescomponents$seasonal
plot(birthstimeseriescomponents)

# --------------------------------------------------------------

# Example 4: Rain fall in London
rain <- scan("http://robjhyndman.com/tsdldata/hurst/precip1.dat",skip=1)
rain
raintimeseries <- ts(rain,start=c(1813), frequency=2)
raintimeseries
plot(raintimeseries)

# Decomposing the rain fall time series
raintimeseriescomponents <- decompose(raintimeseries)
raintimeseriescomponents$seasonal
plot(raintimeseriescomponents)


# --------------------------------------------------------------

# Example 5: Skirts
skirts <- scan("http://robjhyndman.com/tsdldata/roberts/skirts.dat",skip=5)
skirts
skirtstimeseries <- ts(skirts,start=c(1866))
skirtstimeseries
plot(skirtstimeseries)

# Decomposing the rain fall time series
skirtstimeseriescomponents <- decompose(skirtstimeseries)
skirtstimeseriescomponents$seasonal
plot(skirtstimeseriescomponents)

# --------------------------------------------------------------
# Example 6
# Monthly sales for a souvenir shop at a beach resort town in Queensland, Australia, 
# for January 1987-December 1993

souvenir <- scan("http://robjhyndman.com/tsdldata/data/fancy.dat")
souvenirtimeseries <- ts(souvenir, frequency=12, start=c(1987,1))
souvenirtimeseries
plot(souvenirtimeseries)
logsouvenirtimeseries <- log(souvenirtimeseries)
plot(logsouvenirtimeseries)

# Decomposing the souvenir time series
souvenirtimeseriescomponents <- decompose(souvenirtimeseries)
souvenirtimeseriescomponents$seasonal
plot(souvenirtimeseriescomponents)

# --------------------------------------------------------------


