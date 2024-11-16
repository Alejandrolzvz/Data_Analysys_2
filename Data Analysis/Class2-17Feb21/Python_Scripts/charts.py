#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:29:13 2021

@author: franciscocantuortiz
"""


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from datetime import datetime


x = np.arange(0,10) 
y = x ^ 2 
#Simple Plot
plt.plot(x,y)

x = np.arange(0,10) 
y = x ^ 2 
#Labeling the Axes and Title
plt.title("Graph Drawing") 
plt.xlabel("Time") 
plt.ylabel("Distance") 
#Simple Plot
plt.plot(x,y)


x = np.arange(0,10) 
y = x ^ 2 
#Labeling the Axes and Title
plt.title("Graph Drawing") 
plt.xlabel("Time") 
plt.ylabel("Distance") 

# Formatting the line colors
plt.plot(x,y,'r')

# Formatting the line type  
plt.plot(x,y,'>') 

x = np.arange(0,10) 
y = x ^ 2 
z = x ^ 3
t = x ^ 4 
# Labeling the Axes and Title
plt.title("Graph Drawing") 
plt.xlabel("Time") 
plt.ylabel("Distance") 
plt.plot(x,y)

#Annotate
plt.annotate(xy=[2,1], s='Second Entry') 
plt.annotate(xy=[4,6], s='Third Entry') 

x = np.arange(0,10) 
y = x ^ 2 
z = x ^ 3
t = x ^ 4 
# Labeling the Axes and Title
plt.title("Graph Drawing") 
plt.xlabel("Time") 
plt.ylabel("Distance") 
plt.plot(x,y)

#Annotate
plt.annotate(xy=[2,1], s='Second Entry') 
plt.annotate(xy=[4,6], s='Third Entry') 
# Adding Legends
plt.plot(x,z)
plt.plot(x,t)
plt.legend(['Race1', 'Race2','Race3'], loc=4) 

x = np.arange(0,10) 
y = x ^ 2 
z = x ^ 3
t = x ^ 4 
# Labeling the Axes and Title
plt.title("Graph Drawing") 
plt.xlabel("Time") 
plt.ylabel("Distance") 
plt.plot(x,y)

#Annotate
plt.annotate(xy=[2,1], s='Second Entry') 
plt.annotate(xy=[4,6], s='Third Entry') 
# Adding Legends
plt.plot(x,z)
plt.plot(x,t)
plt.legend(['Race1', 'Race2','Race3'], loc=4) 

#Style the background
plt.style.use('fast')
plt.plot(x,z)

df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
df.plot.box(grid='True')

data=[{2,3,4,1},{6,3,5,2},{6,3,5,4},{3,7,5,4},{2,8,1,5}]
Index= ['I1', 'I2','I3','I4','I5']
Cols = ['C1', 'C2', 'C3','C4']
df = DataFrame(data, index=Index, columns=Cols)

plt.pcolor(df)
plt.show()

df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
df.plot.scatter(x='a', y='b')

#create data
x = np.random.rand(40)
y = np.random.rand(40)
z = np.random.rand(40)
colors = np.random.rand(40) 
# use the scatter function
plt.scatter(x, y, s=z*1000,c=colors)
plt.show()


chart = plt.figure()
chart3d = chart.add_subplot(111, projection='3d')

# Create some test data.
X, Y, Z = axes3d.get_test_data(0.08)

# Plot a wireframe.
chart3d.plot_wireframe(X, Y, Z, color='r',rstride=15, cstride=10)

plt.show()
 

#data = pd.read_csv('path_to_file/stock.csv')
#df = pd.DataFrame(data, columns = ['ValueDate', 'Price'])

# Set the Date as Index
#df['ValueDate'] = pd.to_datetime(df['ValueDate'])
#df.index = df['ValueDate']
#del df['ValueDate']


#df.plot(figsize=(15, 6))
#plt.show()
