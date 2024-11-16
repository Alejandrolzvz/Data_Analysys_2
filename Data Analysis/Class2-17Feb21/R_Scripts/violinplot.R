# Violin Plots
install.packages("vioplot")
n
library("vioplot")
x1 <- mtcars$mpg[mtcars$cyl==4]
x2 <- mtcars$mpg[mtcars$cyl==6]
x3 <- mtcars$mpg[mtcars$cyl==8]
vioplot(x1, x2, x3, names=c("4 cyl", "6 cyl", "8 cyl"),
        col="gold")
title("Violin Plots of Miles Per Gallon")

# Second example
ToothGrowth$dose <- as.factor(ToothGrowth$dose)
head(ToothGrowth)

library(ggplot2)
# Basic violin plot
p <- ggplot(ToothGrowth, aes(x=dose, y=len)) + 
  geom_violin()
p
# Rotate the violin plot
p + coord_flip()
# Set trim argument to FALSE
ggplot(ToothGrowth, aes(x=dose, y=len)) + 
  geom_violin(trim=FALSE)

p + scale_x_discrete(limits=c("0.5", "2"))

# violin plot with mean points
p + stat_summary(fun.y=mean, geom="point", shape=23, size=2)

# violin plot with median points
p + stat_summary(fun.y=median, geom="point", size=2, color="red")

p + geom_boxplot(width=0.1)

# violin plot with mean and standard deviation
p <- ggplot(ToothGrowth, aes(x=dose, y=len)) + 
  geom_violin(trim=FALSE)
p + stat_summary(fun.data="mean_sdl", mult=1, 
                 geom="crossbar", width=0.2 )
p + stat_summary(fun.data=mean_sdl, mult=1, 
                 geom="pointrange", color="red")

# Function to produce summary statistics (mean and +/- sd)
data_summary <- function(x) {
  m <- mean(x)
  ymin <- m-sd(x)
  ymax <- m+sd(x)
  return(c(y=m,ymin=ymin,ymax=ymax))
}

p + stat_summary(fun.data=data_summary)

# violin plot with dot plot
p + geom_dotplot(binaxis='y', stackdir='center', dotsize=1)
# violin plot with jittered points
# 0.2 : degree of jitter in x direction
p + geom_jitter(shape=16, position=position_jitter(0.2))

# Change violin plot line colors by groups
p<-ggplot(ToothGrowth, aes(x=dose, y=len, color=dose)) +
  geom_violin(trim=FALSE)
p

# Use custom color palettes
p+scale_color_manual(values=c("#999999", "#E69F00", "#56B4E9"))
# Use brewer color palettes
p+scale_color_brewer(palette="Dark2")
# Use grey scale
p + scale_color_grey() + theme_classic()

# Use single color
ggplot(ToothGrowth, aes(x=dose, y=len)) +
  geom_violin(trim=FALSE, fill='#A4A4A4', color="darkred")+
  geom_boxplot(width=0.1) + theme_minimal()
# Change violin plot colors by groups
p<-ggplot(ToothGrowth, aes(x=dose, y=len, fill=dose)) +
  geom_violin(trim=FALSE)
p

# Use custom color palettes
p+scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))
# Use brewer color palettes
p+scale_fill_brewer(palette="Dark2")
# Use grey scale
p + scale_fill_grey() + theme_classic()

p + theme(legend.position="top")
p + theme(legend.position="bottom")
p + theme(legend.position="none") # Remove legend

p + scale_x_discrete(limits=c("2", "0.5", "1"))

# Change violin plot colors by groups
ggplot(ToothGrowth, aes(x=dose, y=len, fill=supp)) +
  geom_violin()
# Change the position
p<-ggplot(ToothGrowth, aes(x=dose, y=len, fill=supp)) +
  geom_violin(position=position_dodge(1))
p

# Add dots
p + geom_dotplot(binaxis='y', stackdir='center',
                 position=position_dodge(1))
# Change colors
p+scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9"))

# Basic violin plot
ggplot(ToothGrowth, aes(x=dose, y=len)) + 
  geom_violin(trim=FALSE, fill="gray")+
  labs(title="Plot of length  by dose",x="Dose (mg)", y = "Length")+
  geom_boxplot(width=0.1)+
  theme_classic()
# Change color by groups
dp <- ggplot(ToothGrowth, aes(x=dose, y=len, fill=dose)) + 
  geom_violin(trim=FALSE)+
  geom_boxplot(width=0.1, fill="white")+
  labs(title="Plot of length  by dose",x="Dose (mg)", y = "Length")
dp + theme_classic()

# Continusous colors
dp + scale_fill_brewer(palette="Blues") + theme_classic()
# Discrete colors
dp + scale_fill_brewer(palette="Dark2") + theme_minimal()
# Gradient colors
dp + scale_fill_brewer(palette="RdBu") + theme_minimal()
