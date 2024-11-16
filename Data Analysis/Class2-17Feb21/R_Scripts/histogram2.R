AirPassengers

hist(AirPassengers, main="Histogram for Air Passengers")

hist(AirPassengers, xlim=c(100,700), ylim=c(0,30))

hist(AirPassengers, xlab="Passengers", ylab="Frequency of Passengers")

hist(AirPassengers, border="blue", col="green")

hist(AirPassengers, breaks=5) 

hist(AirPassengers, breaks=c(100, 300, 500, 700)) 

hist(AirPassengers, breaks=c(100, seq(200,700, 150)))

hist(AirPassengers, 
     main="Histogram for Air Passengers", 
     xlab="Passengers", 
     border="blue", 
     col="green",
     xlim=c(100,700),
     las=1, 
     breaks=5)

hist(AirPassengers, 
     main="Histogram for Air Passengers", 
     xlab="Passengers", 
     border="blue", 
     col="green", 
     xlim=c(100,700), 
     las=1, 
     breaks=5, 
     prob = TRUE)

lines(density(AirPassengers))
