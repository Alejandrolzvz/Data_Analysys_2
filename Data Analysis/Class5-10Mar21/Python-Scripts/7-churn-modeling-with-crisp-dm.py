# Kernel's Methodolgy

#In this kernel aims to find most suitable model via CRISP-DM strategy for 
# Bank customer which could churn. CRISP-DM is basically data mining 
# methodology but nowadays it use to data science project. Although 
# different approaches have been developed in the field of data science 
# over the years, at the last point reached, where a data science project 
# can be started, which steps should be followed, the outputs of the phases 
# of the project and the measurable steps during the project can be managed 
# with the method shortened as CRISP-DM.

# What is CRISP-DM
#CRISP-DM (Cross Industry Standard Process for Data Mining) 
#    1. Business Understanding: This is the understanding of the business 
# and the understanding of the business being processed.
#    2. Data Understanding: It is the phase of having information about 
# the data structure.
#    3. Data Preparation: This is the data preparation phase.
#    4. Modeling: Creating a model with data is the stage.
#    5. Evaluation: This is the evaluation phase of the model.
#    6. Deployment: Application is the phase of action. After the model 
# is created, the application is started by programming.
    
## 1. Business Understanding
# Basically, expectation of the bank, which customer could be churn and 
# how modelling data of customer of the bank. In line with this expectation, 
# main objective detects customers that could be leave from there.

## 2. Data Understanding
#First of all importing all libraries

#"""Data Preparation Library"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#"""Models Library"""
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

#"""Model Evaluation"""
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#"""Other"""
import os
import warnings
#from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
%matplotlib inline
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category = ConvergenceWarning)

#Importing dataset

# Importing dataset
dt = pd.read_csv("Churn_Modelling.csv")

# First 5 rows of data
dt.head()

# This dataset belongs to the bank that is hidden its name because of data security. The dataset consists of 13 attributes and 10,000 rows. The following shows description of attributes.
# 1.	Customer ID: This attribute is unique and assume that primary key
# 2.	Surname: it belongs to surname of customer and string values
# 3.	Geography: it shows country of customer 
# 4.	Gender: male/female 
# 5.	Credit Score: it gives credit score of customers. That score calculates interbank system. High score shows that the customer debt high repayment capacity. 
# 6.	Age: age of customers
# 7.	Tenure: The number of ages the customer is in the bank.
# 8.	Balance: Customer's money in the bank.
# 9.	Number of Products: Number of products owned by the customer.
# 10.	Credit Card: Whether the customer has a credit card
# 11.	Active Status: Customer's presence in the bank
# 12.	Estimated Salary: Customer's estimated salary
# 13.	Exited: Churn or not

# Remove unneeded columns which are RowNumber, CustomerId and Surnmae


dt = dt.drop(columns=["RowNumber","CustomerId","Surname"])
dt.head()

# There is unique 10.000 customers. Geography and gender are categorical variables. Geography consist of France (%50), Germany (%25) and Spain (%25). Also, gender contain %54 male and %46 female.

# Basic description for continuous variables

dt.describe()

# Above table (basic statical description)shows descriptive statistics of continuous variables from original data. Credit score is between 350 and 850. The average age of the customers is 39. Tenure is maximum ten years which assume age of the bank. Minimum value and first quantiles of Balance are equal 0 which means the distribution may not be normal.

# Missing values checking

pd.DataFrame(dt.isnull().sum(),columns=["Count"])

#Exited column that is flag to define for customers whether churn (%80) or not (%20). Thus, target feature is determined. Other features allow the predictor to classify the value of the target variable. For this reason, the relationship between the target column and other columns is examined in the following visualizations 

# Exited -- CreditScore
sns.violinplot( x=dt["Exited"], y=dt["CreditScore"], linewidth=5)
plt.title("Credit Score Distribution of Churn (Exited)")
plt.show()

# CreditScore boxplot
dt[["CreditScore"]].boxplot();

# Exited -- Age
sns.violinplot( x=dt["Exited"], y=dt["Age"], linewidth=5)
plt.title("Age of Customers Distribution of Churn (Exited)")
plt.show()

# Age boxplot
dt[["Age"]].boxplot();

# Exited -- Tenure
sns.violinplot( x=dt["Exited"], y=dt["Tenure"], linewidth=5)
plt.title("Tenure of Customers Distribution of Churn (Exited)")
plt.show()

# Tenure boxplot
dt[["Tenure"]].boxplot();

# Exited -- Balance
sns.violinplot( x=dt["Exited"], y=dt["Balance"], linewidth=5)
plt.title("Balance of Customers Distribution of Churn (Exited)")
plt.show()

# Balance boxplot
dt[["Balance"]].boxplot();

# Exited -- NumOfProducts
sns.violinplot( x=dt["Exited"], y=dt["NumOfProducts"], linewidth=5)
plt.title("Number of Products of Customers Distribution of Churn (Exited)")
plt.show()

# NumOfProducts boxplot
dt[["NumOfProducts"]].boxplot();

# Exited -- EstimatedSalary
sns.violinplot( x=dt["Exited"], y=dt["EstimatedSalary"], linewidth=5)
plt.title("Estimated Salary of Customers Distribution of Churn (Exited)")
plt.show()

# EstimatedSalary boxplot
dt[["EstimatedSalary"]].boxplot();

#Above figures shows the relationship between six continuous variables and target variable in the form of violin graph. Balance, Tenure, Estimated Salary and Credit score almost appear to be irregular for both churn and not churn. Customers who churn higher age than other. Churn customers When the product numbers are examined; could be interpreted by looking at the graph that customers reduce their products before leaving.

#The following figure in the correlation between the six variables, there is no significant value between any two variables. Only a negative relationship exists between Balance and Number of Product 


# Correlation Matrix
correlationColumns = dt[["CreditScore","Age","Tenure"
    ,"Balance","NumOfProducts","EstimatedSalary"]]

sns.set()
corr = correlationColumns.corr()
ax = sns.heatmap(corr
                 ,center=0
                 ,annot=True
                 ,linewidths=.2
                 ,cmap="YlGnBu")
plt.show()

## 3. DATA PREPARATION
#Since there is a target variable in data of the Bank, classification is made by following the supervised learning method. First, to define which target variable is the model, the target variable and the other variables are separated from each other (Exited and other). Customer ID, Row Number and Surname variables are excluded from the data set because they cannot be input for the model,

# Decomposition predictors and target
predictors = dt.iloc[:,0:10]
target = dt.iloc[:,10:]

#The characters in the gender variable are replaced with 0 or 1. 

try:
    predictors['isMale'] = predictors['Gender'].map({'Male':1, 'Female':0})
except:
    pass

#Dummy variables was reconstructed as 1 or 0 for the three values in the Geography data. Therefore, three different variables were formed. However, the third variable was excluded from the data since two variables included in all three cases. 

try:
    # Geography one shot encoder
    predictors[['France', 'Germany', 'Spain']] = pd.get_dummies(predictors['Geography'])
    # Removal of unused columns.
    predictors = predictors.drop(columns=['Gender','Geography','Spain'])
except:
    pass

# Modelling preparation applies transformation methodology. Three variables 
# (Credit Score, Estimated Salary and Balance) were transformed by 
# normalizing. All values in the variables are represented between 1 and 0.

normalization = lambda x:(x-x.min()) / (x.max()-x.min())
transformColumns = predictors[["Balance","EstimatedSalary","CreditScore"]]
predictors[["Balance","EstimatedSalary","CreditScore"]] = normalization(transformColumns)

# All Predictors Columns
predictors.describe()

#In order to measure the accuracy rate in the modeling, the data set was divided into test and train.

# Train and test splitting
x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size=0.25, random_state=0)
pd.DataFrame({"Train Row Count":[x_train.shape[0],y_train.shape[0]],
              "Test Row Count":[x_test.shape[0],y_test.shape[0]]},
             index=["X (Predictors)","Y (Target)"])

#After all these preparations, the dataset is made ready for modeling which is another step of CRISP-DM method.

## 4. MODELING
#After the pre-processing of the data, one or multiple specific modelling techniques, which are connected to the data mining goal, are selected and data could be modelled. In order to test cogency and the quality of the model, a procedure should be created before the model is built.  Afterward, in order to produce one or more models, the modelling tool could start running on the ready set of data. 

# Numpy excaptions handle
y_train = y_train.values.ravel()

# Decision Tree
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred_dtc = dtc.predict(x_test)
dtc_acc = accuracy_score(y_test,y_pred_dtc)
dtc_acc

# Logistic Regression
logr = LogisticRegression()
logr.fit(x_train,y_train)
y_pred_logr = logr.predict(x_test)
logr_acc = accuracy_score(y_test,y_pred_logr)
logr_acc

# Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred_gnb = gnb.predict(x_test)
gnb_acc = accuracy_score(y_test,y_pred_gnb)
gnb_acc

# K Neighbors Classifier
knn = KNeighborsClassifier( metric='minkowski')
knn.fit(x_train,y_train)
y_pred_knn = knn.predict(x_test)
knn_acc = accuracy_score(y_test,y_pred_knn)
knn_acc

# Random Forrest
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred_rfc = rfc.predict(x_test)
rfc_acc = accuracy_score(y_test,y_pred_rfc)
rfc_acc

# Neural Network
nnc = MLPClassifier()
nnc.fit(x_train,y_train)
y_pred_nnc = nnc.predict(x_test)
nnc_acc = accuracy_score(y_test,y_pred_nnc)
nnc_acc

# Xgboost Classifier
xgboast = xgb.XGBClassifier()
xgboast.fit(x_train, y_train)
xgboast = xgboast.score(x_test,y_test)
xgboast

pd.DataFrame({"Algorithms":["Decision Tree","Logistic Regression","Naive Bayes","K Neighbors Classifier","Random Ferest","Neural Network","Xgboost Classifier"],
              "Scores":[dtc_acc,logr_acc,gnb_acc,knn_acc,rfc_acc,nnc_acc,xgboast]})

# When the accuracy of all models is compared, it is seen that XGBoost 
# algorithm is higher (%86)

## 5. EVALUATION
# In the evaluation stage, the obtained model obtained should evaluated more carefully and the steps while building the model should review in order to be sure that the model appropriately achieves the business objectives.

# Cross validation test
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RFC', RandomForestClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('xgboast', XGBClassifier()))

# evaluate each model in turning kfold results
results_boxplot = []
names = []
results_mean = []
results_std = []
p,t = predictors.values, target.values.ravel()
for name, model in models:
    cv_results = cross_val_score(model, p,t, cv=10)
    results_boxplot.append(cv_results)
    results_mean.append(cv_results.mean())
    results_std.append(cv_results.std())
    names.append(name)
pd.DataFrame({"Algorithm":names,
                                "Accuracy Mean":results_mean,
                                "Accuracy":results_std})

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results_boxplot)
ax.set_xticklabels(names)
plt.show()

# Using the cross-validation method, different train and test sets were 
# created and the model re-run iteratively and the result was increased 
# from 86% to 87%. The accuracy of different algorithms was compared with 
# the application of cross validation method.

# Grid Seach for XGboast
params = {
        'min_child_weight': [1, 2, 3],
        'gamma': [1.9, 2, 2.1, 2.2],
        'subsample': [0.4,0.5,0.6],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3,4,5]
        }
gd_sr = GridSearchCV(estimator=XGBClassifier(),
                     param_grid=params,
                     scoring='accuracy',
                     cv=5,
                     )
gd_sr.fit(predictors, target.values.ravel())
best_parameters = gd_sr.best_params_
pd.DataFrame(best_parameters.values(),best_parameters.keys(),columns=["Best Parameters"])

print("Best score is: ",gd_sr.best_score_)
# Best score is:  0.8674

## 6. DEPLOYMENT
# The deployment phase requires the consequences of the evaluation to verify a strategy for deployment within a particular company. When the results of the project will be used widely, it is significant that the business should take required actions to use definitely the models. At this phase, final report and presentation of the found results are produced.

# If you saved model, you can use Pickle file.
# Pickle cound use "import pickle"

## CONCLUSION
#As a result, using CRISP-DM method, the data set was handled through 
# various processes. This continued at each stage by completing the previous
# stage. XGBoost has the best accuracy of 87%.