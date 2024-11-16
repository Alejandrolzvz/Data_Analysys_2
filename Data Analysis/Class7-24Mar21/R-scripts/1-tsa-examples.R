setwd("/Users/franciscocantuortiz/R")

# Example 1
# Age of Death of Successive Kings of England
# starting with William the Conqueror
# Source: McNeill, "Interactive Data Analysis"getwd()
kings <- scan("http://robjhyndman.com/tsdldata/misc/kings.dat",skip=3)
kings
kingstimeseries <- ts(kings)
kingstimeseries
plot.ts(kingstimeseries)

# Decomposing the kings TS
kingstimeseriescomponents <- decompose(kingstimeseries)
# We get an error because we need at least two periods in the series

# Try at least two cycles
kingstimeseries <- ts(kings, frequency=2)
kingstimeseries
plot.ts(kingstimeseries)

# Decomposing the kings TS
kingstimeseriescomponents <- decompose(kingstimeseries)
kingstimeseriescomponents$seasonal
plot(kingstimeseriescomponents)

# Kings Seasonally Adjusting
kingstimeseriesseasonallyadjusted <- kingstimeseries - kingstimeseriescomponents$seasonal
plot(kingstimeseriesseasonallyadjusted)


# Example 2: Rain
rain <- scan("http://robjhyndman.com/tsdldata/hurst/precip1.dat",skip=1)
rain
rainseries <- ts(rain,start=c(1813), frequency=2)
plot.ts(rainseries,)

# Decomposing the rain TS
rainseriescomponents <- decompose(rainseries)
rainseriescomponents$seasonal
plot(rainseriescomponents)

# Rain Seasonally Adjusting
rainseriesseasonallyadjusted <- rainseries - rainseriescomponents$seasonal
plot(rainseriesseasonallyadjusted)

# --------------------------------------------------------------

# Example 3: Skirts
skirts <- scan("http://robjhyndman.com/tsdldata/roberts/skirts.dat",skip=5) 
skirtseries <- ts(skirts,start=c(1866))
plot.ts(skirtseries)

# --------------------------------------------------------------

# Example 4: Covid-19 Confirmed casews
mydata <- read.csv("~/R/novel-corona-virus-2019-dataset/covid_19_data.csv")
mydata
confirmed<-c(444,	444,	549,	761,	1058,	1423,	3554,	3554,	4903,	5806,	7153,	11177,	13522,	16678,	19665,	22112,	24953,	27100,	29631,	31728,	33366,	33366,	48206,	54406,	56249,	58182,	59989,	61682,	62031,	62442,	62662,	64084,	64084,	64287,	64786,	65187,	65596,	65914,	66337,	66907,	67103,	67217,	67332,	67466,	67592,	67666,	67707,	67743,	67760,	67773,	67781,	67786,	67790,	67794,	67798,	67799,	67800,	67800,	67800,	67800,	67800,	67800,	67801,	67801,	67801,	67801,	67801,	67801,	67801)
confirmed
confts<-ts(confirmed, frequency=4)
confts
plot.ts(confts)

# Decomposing the births TS
conftimeseriescomponents <- decompose(confts)
conftimeseriescomponents$seasonal
plot(conftimeseriescomponents)


# --------------------------------------------------------------

# Example 4
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

# --------------------------------------------------------------

# Example 5
# Monthly sales for a souvenir shop at a beach resort town in Queensland, Australia, 
# for January 1987-December 1993
souvenir <- scan("http://robjhyndman.com/tsdldata/data/fancy.dat")
souvenirtimeseries <- ts(souvenir, frequency=12, start=c(1987,1))
souvenirtimeseries
plot.ts(souvenirtimeseries)
logsouvenirtimeseries <- log(souvenirtimeseries)
plot.ts(logsouvenirtimeseries)

# Decomposing the souvenir TS
souvenirtimeseriescomponents <- decompose(souvenirtimeseries)
souvenirtimeseriescomponents$seasonal
plot(souvenirtimeseriescomponents)

# Souvenir Seasonally Adjusting
souvenirtimeseriesseasonallyadjusted <- souvenirtimeseries - souvenirtimeseriescomponents$seasonal
plot(souvenirtimeseriesseasonallyadjusted)


A1 <- ts(runif(24, 1, 100), frequency=12)
decompose(A1)
plot.ts(A1)
