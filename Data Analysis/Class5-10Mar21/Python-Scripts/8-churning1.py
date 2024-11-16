#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:10:55 2021

@author: franciscocantuortiz
"""

# Source: https://github.com/mmcuri/ds_handson/blob/master/Telecom_Churn_Prediction.ipynb



# Open In Colab
# Churn Prediction


# Companies usually have a greater focus on customer acquisition and keep retention as a secondary priority. However, it can cost five times more to attract a new customer than it does to retain an existing one. Increasing customer retention rates by 5% can increase profits by 25% to 95%, according to research done by Bain & Company.

# Churn is a measurement of business that shows customers who stop doing business with a company or a service, also known as customer attrition. By following this metric, what most businesses could do was try to understand the reason behind churn numbers and tackle those factors, with reactive action plans. 

# But what if you could know in advance that a specific customer is likely to leave your business, and have a chance to take proper actions in time to prevent it from happening? The reasons that lead customers to the cancellation decision can be numerous, coming from poor service quality, delay on customer support, prices, new competitors entering the market, and so on. Usually, there is no single reason, but a combination of events that culminated in customer displeasure.

# If your company were not capable to identify these signals and take actions prior to the cancel button click, there is no turning back, your customer is already gone. But you still have something valuable: the data. Your customer left very good clues about where you left to be desired. It can be a valuable source for meaningful insights and to train customer churn models. Learn from the past, and have strategic information at hand to improve future experiences, it's all about machine learning.

# When it comes to the telecommunications segment, there is great room for opportunities. The wealth and the amount of customer data that carriers collect can contribute a lot to shift from a reactive to a proactive position. The emergence of sophisticated artificial intelligence and data analytics techniques further help leverage this rich data to address churn in a much more effective manner. In this article, I'm going to use a customer base dataset from an anonymous carrier, made available by the platform IBM Developer.

# The main goal is to develop a machine learning model capable to predict customer churn based on the customer's data available. I will use mainly Python, Pandas, and Scikit-Learn libraries for this implementation. The complete code you can find on my GitHub. To accomplish that, we will go through the below steps:

# Exploratory analysis
# Data preparation
# Train, tune and evaluate machine learning models
# Project Initial Setup

# install scikit plot package
#!pip install scikit-plot

# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt
#%matplotlib inline
#!pip install imblearn
from imblearn.under_sampling import RandomUnderSampler

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report  
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier


# configure graph style
sns.set_style('darkgrid')
#palette=sns.color_palette("GnBu_d")
#palette=sns.color_palette("BuGn_r")
#palette = sns.light_palette("navy", reverse=True)
palette = sns.color_palette("coolwarm", 7)
import pandas.util.testing as tm

# The Data
# This dataset contains a total of 7,043 customers and 21 attributes, 
# coming from personal characteristics, services signatures, and contract 
# details. Out of the entries, 
# 5,174 are active customers and 1,869 are churned, which demonstrates 
# that the dataset is highly unbalanced. 
# The target variable for this assessment is going to be the feature Churn.

# Importing dataset

# Loading dataset
df = pd.read_csv('churning.csv')

# check first 5 entries
df.head()

# Data Dictionary
# customerID - Custumer unique identifier
# gender - Customer gender - ['Female' 'Male']
# SeniorCitizen - Elderly or retired person, a senior citizen is someone 
##   who has at least attained the age of 60 of 65 years
# Partner - - ['No' 'Yes']
# Dependents - If customer has dependents - ['No' 'Yes']
# Tenure - Customer lifespan (in months)
# PhoneService - - ['No' 'Yes']
# MultipleLines - - ['No' 'No phone service' 'Yes']
# InternetService - - ['No' 'No internet service' 'Yes']
# OnlineSecurity - - ['No' 'No internet service' 'Yes']
# OnlineBackup - - ['No' 'No internet service' 'Yes']
# DeviceProtection - - ['No' 'No internet service' 'Yes']
# TechSupport - - ['No' 'No internet service' 'Yes']
# StreamingTV - - ['No' 'No internet service' 'Yes']
# StreamingMovies - - ['No' 'No internet service' 'Yes']
# Contract - Type of contract - ['Month-to-month' 'One year' 'Two year']
# PaperlessBilling - - ['No' 'Yes']
# PaymentMethod - payment method - ['Bank transfer (automatic)', 
## 'Credit card (automatic)', 'Electronic check', 'Mailed check']
# MonthlyCharges - Monthly Recurring Charges
# TotalCharges - Life time value
# Churn - Churn value, the targer vector - ['No' 'Yes']

# Dataframe size and info

def get_df_size(df, header='Dataset dimensions'):
  print(header,
        '\n# Attributes: ', df.shape[1], 
        '\n# Entries: ', df.shape[0],'\n')
  
get_df_size(df)
# Dataset dimensions 
# Attributes:  21 
# Entries:  7043 

# Features and data types
# The feature TotalCharges got read by Pandas as object data type. This have impacts during the exploratory analysis and have to be handled. We will convert datatype to float64 in the coming sections.


df.info()


# Exploratory Analysis
# Checking missing values
# Before checking the missing values, we are going to replace all the blank spaces ocurreces that this dataset might have.

# replacing all the blank values with NaN 
df_clean = df.replace(r'^\s*$', np.nan, regex=True)

# print missing values
print("Missing values (per feature): \n{}\n".format(df_clean.isnull().sum()))

# After that we can see that feature TotalCharges has 11 missing values. We are going to replace these missing values by the TotalCharges median.

total_charges_median = df_clean.TotalCharges.median()
df_clean['TotalCharges'].fillna(total_charges_median, inplace=True)
# Converting data types
# While importing dataset, Pandas read the column TotalCharges as object because it had some entries populated with blank spaces instead of NaN value. For the analysis we will convert datatype of this feature from object to float64.

df_clean['TotalCharges'] = df_clean['TotalCharges'].apply(pd.to_numeric)

# Unique values per feature
# By checking feature's unique values we can see that the column customerID have unique identifiers for each customer, which confirms that each row represents a single customer. This feature does not contribute for this analysis, therefore we are going to drop the column.

print("Unique values (per feature): \n{}\n".format(df.nunique()))

df_clean = df_clean.drop('customerID', axis=1)
# Descriptive statistics

df_clean.describe()

fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(8,5))

sns.boxplot(df_clean['MonthlyCharges'], ax=ax[0])
sns.boxplot(df_clean['TotalCharges'], ax=ax[1])

plt.tight_layout()

# Dataset features and their values
# This is a very important information to help us to undesrstand the dataset 
# will be working with. Few observations:

# Feature SeniorCitizen is binary, entries have value 1 for Yes and 0 for No
# Feature Tenure has the max value in 72, which can indicate that this 
# service provider has maximum of 6 years
# The only features that are not categorical are Monthly Charges and 
# TotalCharges, all the remaining are categorical kinds

features_obj = df_clean.columns

for f in features_obj:
  print(f)
  print(np.unique(df_clean[f].values))


# Customer lifespan
# Helper Functions

def display_percent(plot, feature, total):
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        x = p.get_x() + p.get_width() / 2
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), size = 10, ha="center")
    plt.show()
    
# How long is the customer lifespan until subscription cancellation?
# The majority of churn rate is observed on the subscription first month, 
# totalling 20,3% of subscribers leave on the first month
# Most of the subscribers leave on the first 3 months, 
# totalling 31.9% of the total churn

p = sns.color_palette("coolwarm", 10)
p.reverse()

df_top_churn = pd.DataFrame(df_clean[df_clean['Churn'] == 'Yes']['tenure'].value_counts().sort_values(ascending=False))
total_churn = df_clean[df_clean['Churn'] == 'Yes'].shape[0]

fig, ax = plt.subplots(figsize=(10,5))
sns_lifespan = sns.barplot( x = df_top_churn[:10].index, y = df_top_churn[:10].tenure, ax=ax, palette=p, order=df_top_churn[:10].index)
plt.xticks(size=12)
plt.xlabel('Customer Lifespan (in months)', size=12)
plt.yticks(size=12)
plt.ylabel('Churn', size=12)
plt.tick_params(labelleft=False)

display_percent(ax, df_top_churn, total_churn)

sns_lifespan.figure.savefig("churn_rate_tenure.png", dpi=600)

# Understanding the profile of churn customers
# Helper Functions

# helper funtion - display count plot
def displayCountPlot(cat_list, df, rows=1, columns=3, figsize=(14,2.5), export=False):
  
  """
    Display countplot based on a set of features

    # Arguments
      cat_list: array, List of features
      df: DataFrame, dataset
      rows: int, number of rows
      columns: int, number of columns
      figsize: figure size, e.g (10, 5)

  """

  fig, ax = plt.subplots(ncols=columns, figsize=figsize)
  
  idx = 0
  for c in cat_list:
    idx += 1
    plt.subplot(rows, columns, idx)
    ax = sns.countplot(x=df[c], data=df, palette=palette)

    plt.xticks(size=10)
    plt.xlabel('')
    plt.yticks(size=12)
    plt.ylabel('')
    plt.subplots_adjust(hspace = 0.4)
    ax.tick_params(labelleft=False)
    ax.set_title(c, alpha=0.8)

    print_rate(ax, df.shape[0])

  if export :
    save_img(fig, ax)

  plt.tight_layout()
  plt.show()

  return fig

def print_rate(ax, total):
  for p in ax.patches:
    text = '{:.1f}% ({})'.format(100 * p.get_height() / total, p.get_height())
    x = p.get_x() + p.get_width() / 2
    y = p.get_y() + p.get_height() * 0.5
    ax.annotate(text, (x, y), size = 10, ha="center", va="center")

def save_img(fig, ax):
  fig.savefig(ax.get_title(), dpi=600)

df_churn = df_clean[df_clean['Churn'] == 'Yes']
df_churn = df_churn.drop('Churn', axis=1)

df_churn.loc[df_churn['SeniorCitizen'] == 0,'SeniorCitizen'] = 'No' 
df_churn.loc[df_churn['SeniorCitizen'] == 1,'SeniorCitizen'] = 'Yes'

personal_attributes = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
services_attributes = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                          'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                          'StreamingMovies']
contract_attributes = ['Contract', 'PaperlessBilling', 'PaymentMethod']

# a) In terms of personal attibutes
# Let's review which personal charactiristic contribute mostly for the 
# cancellation decision. From the available dataset those are:

# Gender
# SeniorCitizen
# Partner
# Dependents

# Below charts can provide some meaningful insights such as:

# Customers without dependents are 4 times more likely to churn
# Senior citizens are 3 times less likely to churn
# Partners are almost 2 times less likely to churn

displayCountPlot(personal_attributes, df_churn, rows=1, columns=4, export=True)

# b) In terms of services
# Let's review which personal charactiristic contributes mostly for 
# the cancellation decision. From the available dataset those are:

# PhoneService
# MultipleLines
# InternetService
# OnlineSecurity
# OnlineBackup
# DeviceProtection
# TechSupport
# StreamingTV
# StreamingMovies

# The below charts show the features where high discrepancies between the 
# classes could be noticed. It gives insights regarding which kind of 
# carrier services the customers that are more likely to defeat make use:

# The majority of customers that cancel their subscription have 
# Phone Service enabled
# Customers that have Fiber-Optic Internet Service are more likely to 
# cancel than those who have DSL
# Customers that do not have Online Security, Device Protection, 
# Online Backup, and Tech Support services enabled are more likely to leave

services_attributes_filtered = ['PhoneService', 'InternetService', 'OnlineSecurity',
                          'OnlineBackup', 'DeviceProtection', 'TechSupport',]

displayCountPlot(services_attributes_filtered, df_churn, rows=3, columns=3, figsize=(14,8), export=True)

# c) In terms of contract aspects
# Let's review which personal charactiristic contributes mostly for the 
# cancellation decision. From the available dataset those are:

# Contract
# PaperlessBilling
# PaymentMethod

# Below charts give insights regarding the contract aspects that can make a 
# subscriber more likely to churn:

# The majority of customers that cancel their subscription have 
# Month-to-month Contract type and Paperless Billing enabled
# Customers that have Payment Method as Eletronic Check are more likely 
# to leave

df_churn['PaymentMethod'] = df_churn['PaymentMethod'].str.replace('(automatic)', '').str.replace('(', '').str.replace(')', '').str.strip()

fig = displayCountPlot(contract_attributes, df_churn, rows=1, columns=3)

fig.savefig("contract.png", dpi=600)

# Imbalanced data
# Column Churn is the target vector to be used to train the ML models. The class No have much more entries then class Yes, which demonstrates that the dataset is highly imbalanced. Ideally the dataset should be balanced to avoid models overfitting.

print(df_clean[df_clean['Churn'] == 'No'].shape[0])
print(df_clean[df_clean['Churn'] == 'Yes'].shape[0])

"""fig, ax = plt.subplots()
sns.countplot(df_clean['Churn'], palette=palette)

plt.xticks(size=12)
plt.xlabel('Churn', size=12)
plt.yticks(size=12)
plt.ylabel('# Customers', size=12)"""

displayCountPlot(['Churn'], df_clean, rows=1, columns=1, figsize=(5,3), export=True)
# 5174
# 1869

# Data Preparation
# Split features into binary, numeric or categorical

binary_feat = df_clean.nunique()[df_clean.nunique() == 2].keys().tolist()
numeric_feat = [col for col in df_clean.select_dtypes(['float','int']).columns.tolist() if col not in binary_feat]
categorical_feat = [ col for col in df_clean.select_dtypes('object').columns.to_list() if col not in binary_feat + numeric_feat ]

df_proc = df_clean.copy()
# Features encoding
# a) Apply label encoding for binary features

le = LabelEncoder()
for i in binary_feat:
  df_proc[i] = le.fit_transform(df_proc[i])
  print(i, '\n', np.unique(df_proc[f].values))


# b) Convert categorical variable into dummy variables

# print(categorical_feat)
df_proc = pd.get_dummies(df_proc, columns=categorical_feat)
print(df_proc.columns)

get_df_size(df, header='Original dataset:')
get_df_size(df_proc, header='Processed dataset:')

df_proc.head()
# Original dataset: 
# Attributes:  21 
# Entries:  7043 

# Processed dataset: 
# Attributes:  41 
# Entries:  7043 

# ---------------------------------------------------

# Split train and test data

# split df_proc in feature matrix and target vector
X=df_proc.drop('Churn', axis=1)
y=df_proc['Churn']

# split df_proc between train and test
X_train, X_test, y_train, y_test = train_test_split(X, y)
# Machine Learning Model
# Helper Functions

# cross-validation function
def val_model(X, y, clf, quite=False):
  """
    Make cross-validation for a given model

    # Arguments
      X: DataFrame, feature matrix
      y: Series, target vector
      clf: classifier from scikit-learn
      quite: bool, indicate if funcion should print the results

    # Returns
      float, validation scores

  """

  X = np.array(X)
  y = np.array(y)

  pipeline = make_pipeline(StandardScaler(), clf)
  scores = cross_val_score(pipeline, X, y, cv=5, scoring='recall')

  if quite == False:
    print("##### ", clf.__class__.__name__, " #####")
    print("scores:", scores)
    print("recall: {:.3f} (+/- {:.2f})".format(scores.mean(), scores.std()))

  return scores.mean()

def getClfRecallScores(X_train, y_train, *clf_list):
  """
  Provides recall score gor a given list of models

  # Arguments
    X_train: X_train
    y_train: y_train
    *clf_list: list of classifiers

  # Returns
    DataFrame, recall scores

  """

  model_name = []
  recall = []

  for model in clf_list:
    model_name.append(model.__class__.__name__)
    recall.append(val_model(X_train, y_train, model))
  
  return pd.DataFrame(data=recall, index=model_name, columns=['Recall']).sort_values(by='Recall', ascending=False)


# -------------------------------------------------------------
  
# Balancing the data

# under sampling
rus = RandomUnderSampler()
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

get_df_size(X_train, header='Before balancing:')
get_df_size(X_train_rus, header='After balancing:')

# make sure the number of classes are equal distibuted
np.unique(y_train_rus, return_counts=True)
# Before balancing: 
# Attributes:  40 
# Entries:  5282 

# After balancing: 
# Attributes:  40 
# Entries:  2842 

# ------------------------------------------------------

# Standardizing the data

# standardizing X_train and X_test
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train_rus = scaler.fit_transform(X_train_rus)
X_test = scaler.transform(X_test)
# Create baseline using Cross Validation

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# instaciate models

dt = DecisionTreeClassifier()
svc = SVC()
lr = LogisticRegression()
xgb = XGBClassifier()

df_scores = getClfRecallScores(X_train_rus, y_train_rus, dt, svc, lr, xgb)

print("Here are the scores:---> ", df_scores)

# --------------------------------------------------------------------

# Tuning Models
# As LogisticRegression and SVC performed better in terms of Recall metric, 
# I'm going to use those to tune the hyperparameters and check if it can 
# deliver even better results.

# Logistic Regression
# I will be tunning solver and C in the Logistic Regression model. 
# As can be seen below it presented a slight improvement after tuned, 
# incresing Recall from 0.80 to 0.82.

# --------------------------------------------------------------------

kfold = StratifiedKFold(n_splits=5, shuffle=True)

lr = LogisticRegression()

param_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear'],
              'C': [0.001, 0.01, 1, 10, 100]}

search = GridSearchCV(lr, param_grid, scoring='recall', cv=kfold)
result = search.fit(X_train_rus, y_train_rus)

print(f'Best recall: {result.best_score_} for {result.best_params_}')

model_lr = LogisticRegression(solver='newton-cg', C=0.001)
model_lr.fit(X_train_rus, y_train_rus)
y_pred_lr = model_lr.predict(X_test)
lr_corr = confusion_matrix(y_test, y_pred_lr, normalize='true')
print(classification_report(y_test, y_pred_lr))

# ---------------------------------------------------------------------

# SVM Model
# I will be tunning kernel and C in the SVM model. SVM had a great 
# improvement in Recall after tunning, increasing Recall 
# from 0.80 to 0.91., which is an excelent score.

# param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#               'C': [0.001, 0.01, 1, 10, 100] }

search = GridSearchCV(SVC(), param_grid, scoring='recall', cv=kfold)
#result = search.fit(X_train_rus, y_train_rus)

print(f'Best recall: {result.best_score_} for {result.best_params_}')

# Best recall: 0.9338547071905114 for {'C': 0.01, 'kernel': 'poly'}
model_svm = SVC(kernel='poly', C=0.01)
model_svm.fit(X_train_rus, y_train_rus)
y_pred_svm = model_svm.predict(X_test)
svm_corr = confusion_matrix(y_test, y_pred_svm, normalize='true')

print(classification_report(y_test, y_pred_svm))

# -------------------------------------------------------------------

# Comparing LR and SVM Model

fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (10,4))
fig.suptitle('Correlation Matrix')

sns.heatmap(svm_corr, annot=True, linewidths=.1, cmap=palette, ax=ax[0])
ax[0].set_title('SVM')
ax[0].set_ylabel('True')
ax[0].set_xlabel('Predicted')

sns.heatmap(lr_corr, annot=True, linewidths=.1, cmap=palette, ax=ax[1])
ax[1].set_title('Logistic Regression')
ax[1].set_ylabel('True')
ax[1].set_xlabel('Predicted')


plt.show()

fig.savefig("correlation_.png", dpi=600)

# Conclusion
# No algorithm will predict churn with 100% accuracy. 
# There will always be a trade-off between precision and recall. 
# That's why it's important to test and understand the strengths and 
# weaknesses of each classifier and get the best out of each. 
# 
# If the goal is to engage and reach out to the customers to prevent them 
# from churning, it's acceptable to engage with those who are mistakenly 
# tagged as 'not churned,' as it does not cause any negative problem. 
# It could potentially make them even happier with the service. 
# This is the kind of model that can add value from day one if proper 
# action is taken out of it.