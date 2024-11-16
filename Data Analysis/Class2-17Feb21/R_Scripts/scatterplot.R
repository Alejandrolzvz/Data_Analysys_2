# Simple Scatterplot
attach(mtcars)
plot(wt, mpg, main="Scatterplot Example",
     xlab="Car Weight ", ylab="Miles Per Gallon ", pch=19)

# Add fit lines
abline(lm(mpg~wt), col="red") # regression line (y~x)
lines(lowess(wt,mpg), col="blue") # lowess line (x,y)

# Enhanced Scatterplot of MPG vs. Weight
# by Number of Car Cylinders
library(car)
scatterplot(mpg ~ wt | cyl, data=mtcars,
            xlab="Weight of Car", ylab="Miles Per Gallon",
            main="Enhanced Scatter Plot",
            labels=row.names(mtcars))

# Basic Scatterplot Matrix
pairs(~mpg+disp+drat+wt,data=mtcars,
      main="Simple Scatterplot Matrix")

# Scatterplot Matrices from the glus Package
library(gclus)
dta <- mtcars[c(1,3,5,6)] # get data
dta.r <- abs(cor(dta)) # get correlations
dta.col <- dmat.color(dta.r) # get colors
# reorder variables so those with highest correlation
# are closest to the diagonal
dta.o <- order.single(dta.r)
cpairs(dta, dta.o, panel.colors=dta.col, gap=.5,
       main="Variables Ordered and Colored by Correlation" )

# High Density Scatterplot with Binning
library(hexbin)
x <- rnorm(1000)
y <- rnorm(1000)
bin<-hexbin(x, y, xbins=50)
plot(bin, main="Hexagonal Binning")

# Second Example

x <- mtcars$wt
y <- mtcars$mpg
# Plot with main and axis titles
# Change point shape (pch = 19) and remove frame.
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
# Add regression line
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
abline(lm(y ~ x, data = mtcars), col = "blue")

# Add loess fit
plot(x, y, main = "Main title",
     xlab = "X axis title", ylab = "Y axis title",
     pch = 19, frame = FALSE)
lines(lowess(x, y), col = "blue")

install.packages("car")
library("car")

scatterplot(wt ~ mpg, data = mtcars)

# Suppress the smoother and frame
scatterplot(wt ~ mpg, data = mtcars, 
            smoother = FALSE, grid = FALSE, frame = FALSE)

# Scatter plot by groups ("cyl")
scatterplot(wt ~ mpg | cyl, data = mtcars, 
            smoother = FALSE, grid = FALSE, frame = FALSE)

# Add labels
scatterplot(wt ~ mpg, data = mtcars,
            smoother = FALSE, grid = FALSE, frame = FALSE,
            labels = rownames(mtcars), id.n = nrow(mtcars),
            id.cex = 0.7, id.col = "steelblue",
            ellipse = TRUE)
# Third <example Iris dataset
head(iris)

# Prepare the data set
x <- iris$Sepal.Length
y <- iris$Sepal.Width
z <- iris$Petal.Length
grps <- as.factor(iris$Species)
# Plot
library(scatterplot3d)
scatterplot3d(x, y, z, pch = 16)

# Change color by groups
# add grids and remove the box around the plot
# Change axis labels: xlab, ylab and zlab
colors <- c("#999999", "#E69F00", "#56B4E9")
scatterplot3d(x, y, z, pch = 16, color = colors[grps],
              grid = TRUE, box = FALSE, xlab = "Sepal length", 
              ylab = "Sepal width", zlab = "Petal length")
