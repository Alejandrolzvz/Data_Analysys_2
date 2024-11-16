#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 22:11:40 2021

@author: franciscocantuortiz
"""

# Start with loading all necessary libraries
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
%matplotlib inline



# Load in the dataframe
df = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)

# Looking at first 5 rows of the dataset
df.head()

print("There are {} observations and {} features in this dataset. \n".format(df.shape[0],df.shape[1]))

print("There are {} types of wine in this dataset such as {}... \n".format(len(df.variety.unique()), ", ".join(df.variety.unique()[0:5])))

print("There are {} countries producing wine in this dataset such as {}... \n".format(len(df.country.unique()), ", ".join(df.country.unique()[0:5])))
df[["country", "description","points"]].head()

# Groupby by country
country = df.groupby("country")

# Summary statistic of all countries
country.describe().head()

# This selects the top 5 highest average points among all 44 countries:
country.mean().sort_values(by="points",ascending=False).head()

# You can plot the number of wines by country using the plot method of Pandas DataFrame and Matplotlib
plt.figure(figsize=(15,10))
country.size().sort_values(ascending=False).plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Country of Origin")
plt.ylabel("Number of Wines")
plt.show()

# Does quantity over quality?# Let's now take a look at the plot of all 44 countries by its highest rated wine, using the same plotting technique as above:
#plt.figure(figsize=(15,10))
#country.max().sort_values(by="points",ascending=False)["points"].plot.bar()
#plt.xticks(rotation=50)
#plt.xlabel("Country of Origin")
#plt.ylabel("Highest point of Wines")
#plt.show()

# Set up a basic WordCloud
?WordCloud

# Start with one review:
text = df.description[0]

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# lower max_font_size, change the maximum number of word and lighten the background:
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Save the image in the img folder:
wordcloud.to_file("img/first_review.png")

text = " ".join(review for review in df.description)
print ("There are {} words in the combination of all review.".format(len(text)))

# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["drink", "now", "wine", "flavor", "flavors"])

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Now, let's pour these words into a cup of wine!
wine_mask = np.array(Image.open("wine_mask.png"))
wine_mask

def transform_format(val):
    if val == 0:
        return 255
    else:
        return val
    
# Transform your mask into a new one that will work with the function:
transformed_wine_mask = np.ndarray((wine_mask.shape[0],wine_mask.shape[1]), np.int32)

for i in range(len(wine_mask)):
    transformed_wine_mask[i] = list(map(transform_format, wine_mask[i]))
    
 
# Check the expected result of your mask
transformed_wine_mask

wc = WordCloud(background_color="white", max_words=1000, mask=transformed_wine_mask, 
               stopwords=stopwords, contour_width=3, contour_color='firebrick')

# Generate a wordcloud
wc.generate(text)

# store to file
wc.to_file("wine.png")

# show
plt.figure(figsize=[20,10])
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


    