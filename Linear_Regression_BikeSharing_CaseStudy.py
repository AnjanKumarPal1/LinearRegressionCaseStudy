#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import of the requisite libraries & packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Read the csv file as given for the case study
bikeSharing_dt = pd.read_csv("C:\\Anjan\\Linear Regression case study\\day.csv")

# Review the head, i.e., first five records
bikeSharing_dt.head()


# In[5]:


# Get the shape
bikeSharing_dt.shape


# In[6]:


# Now check the dataframe for null and datatypes 
bikeSharing_dt.info()


# In[8]:


# Review the details of numeriacl data
bikeSharing_dt.describe()


# In[9]:


# Review the columns of data
bikeSharing_dt.columns


# In[10]:


# Find the size of data
bikeSharing_dt.size


# In[22]:


# Find out the dimensions of data
bikeSharing_dt.ndim
bikeSharing_dt.values


# In[39]:


# Cleaning of the data those are not useful for the model building and EDA
import datetime
bikeSharing_dt['dteday'] = pd.to_datetime(bikeSharing_dt['dteday'], dayfirst=True)
bikeSharing_dt['dteday'] = pd.to_datetime(bikeSharing_dt['dteday'], format='%Y-%m-%d %H:%M:%S').dt.strftime('%d-%m-%Y')
bikeSharing_dt.values


# In[41]:


# Before dropping date, we introduce a days_old variable which indicates how old is the business
bikeSharing_dt['days_old'] = (pd.to_datetime(bikeSharing_dt['dteday'],format= '%d-%m-%Y') - pd.to_datetime('01-01-2018',format= '%d-%m-%Y')).dt.days
bikeSharing_dt.head()


# In[43]:


# Droping instant column as it is index column which has nothing to do with target
bikeSharing_dt.drop(['instant'], axis = 1, inplace = True)

# Dropping dteday as we have already have month and weekday columns to work with
bikeSharing_dt.drop(['dteday'], axis = 1, inplace = True)

# Dropping casual and registered columnsa as as we have cnt column which is sum of the both that is the target column

bikeSharing_dt.drop(['casual'], axis = 1, inplace = True)
bikeSharing_dt.drop(['registered'], axis = 1, inplace = True)


# In[45]:


# Now check the data frame after dropping
bikeSharing_dt.head()


# In[46]:


# Now find out the correlation
bikeSharing_dt.corr()


# In[ ]:


# We can see that features like season, mnth, weekday and weathersit are integers although they should be non-numerical categories.


# In[47]:


# To check whether there is any missing value for the columns
#Print null counts by column
bikeSharing_dt.isnull().sum()


# In[ ]:


# So there are no null values


# In[48]:


# To check if there is any Outlier
bikeSharing_dt.columns


# In[49]:


#Print number of unique values in all columns
bikeSharing_dt.nunique()


# In[50]:


# Draw box plots for indepent variables with continuous values
cols = ['temp', 'atemp', 'hum', 'windspeed']
plt.figure(figsize=(18,4))

i = 1
for col in cols:
    plt.subplot(1,4,i)
    sns.boxplot(y=col, data=bikeSharing_dt)
    i+=1


# In[52]:


# The above plot shows that there are no outliers
# Now we will do an EDA on this dataframe
# It's time to Convert season and weathersit to categorical types

bikeSharing_dt.season.replace({1:"spring", 2:"summer", 3:"fall", 4:"winter"},inplace = True)

bikeSharing_dt.weathersit.replace({1:'good',2:'moderate',3:'bad',4:'severe'},inplace = True)

bikeSharing_dt.mnth = bikeSharing_dt.mnth.replace({1: 'jan',2: 'feb',3: 'mar',4: 'apr',5: 'may',6: 'jun',
                  7: 'jul',8: 'aug',9: 'sept',10: 'oct',11: 'nov',12: 'dec'})

bikeSharing_dt.weekday = bikeSharing_dt.weekday.replace({0: 'sun',1: 'mon',2: 'tue',3: 'wed',4: 'thu',5: 'fri',6: 'sat'})
bikeSharing_dt.head()


# In[53]:


#Draw pairplots for continuous numeric variables using seaborn
plt.figure(figsize = (15,30))
sns.pairplot(data=bikeSharing_dt,vars=['cnt', 'temp', 'atemp', 'hum','windspeed'])
plt.show()


# In[54]:


# Observations:
# 1. Looks like the temp and atemp has the highest corelation with the target variable cnt
# 2. temp and atemp are highly co-related with each other
# 3. As seen from the correlation map, output variable has a linear relationship with variables like temp, atemp.

# Checking continuous variables relationship with each other
sns.heatmap(bikeSharing_dt[['temp','atemp','hum','windspeed','cnt']].corr(), cmap='BuGn', annot = True)
plt.show()


# In[56]:


# Here we see that temp and atemp has correlation more than .99 means almost 1 (highly correlated)
# and atemp seems to be derived from temp so atemp field can be dropped here only
# Now Draw Heatmap of correlation between variables
corr = bikeSharing_dt.corr()
plt.figure(figsize=(25,10))

#Draw Heatmap of correlation
sns.heatmap(corr,annot=True, cmap='YlGnBu' )
plt.show()


# In[ ]:


# From the correlation map, temp, atemp and days_old seems to be highly correlated 
# and only should variable can be considered for the model. 
# However let us elminate it based on the Variance Inflation Factor later during the model building.
# We also see Target variable has a linear relationship with some of the indeptendent variables. 
# Good sign for building a linear regression Model.

# Now Analysing Categorical Variabels with target variables


# In[58]:


# Boxplot for categorical variables to see demands
vars_cat = ['season','yr','mnth','holiday','weekday','workingday','weathersit']
plt.figure(figsize=(15, 15))
for i in enumerate(vars_cat):
    plt.subplot(3,3,i[0]+1)
    sns.boxplot(data=bikeSharing_dt, x=i[1], y='cnt')
plt.show()


# Observations:
# Here many insights can be drawn from these plots
# 
# 1. Season3:fall has highest demand for rental bikes
# 2. I see that demand for next year has grown
# 3. Demand is continuously growing each month till June. September month has highest demand. After September, demand is decreasing
# 4. When there is a holiday, demand has decreased.
# 5. Weekday is not giving clear picture abount demand.
# 6. The clear weathershit has highest demand
# 7. During September, bike sharing is more. During the year end and beginning, it is less, could be due to extereme weather conditions.
# 

# In[60]:


plt.figure(figsize=(6,5),dpi=110)
plt.title("Cnt vs Temp",fontsize=16)
sns.regplot(data=bikeSharing_dt,y="cnt",x="temp")
plt.xlabel("Temperature")
plt.show()


# Observation:
# Demand for bikes is positively correlated to temp.
# We can see that cnt is linearly increasing with temp indicating linear relation.

# In[61]:


plt.figure(figsize=(6,5),dpi=110)
plt.title("Cnt vs Hum",fontsize=16)
sns.regplot(data=bikeSharing_dt,y="cnt",x="hum")
plt.xlabel("Humidity")
plt.show()


# Observation:
# Hum is values are more scattered around.
# Although we can see cnt decreasing with increase in humidity.

# In[63]:


plt.figure(figsize=(6,5),dpi=110)
plt.title("Cnt vs Windspeed",fontsize=16)
sns.regplot(data=bikeSharing_dt,y="cnt",x="windspeed")
plt.show()


# Observations:
# Windspeed is values are more scattered around.
# Although we can see cnt decreasing with increase in windspeed.

# In[64]:


# Get all the statistics
bikeSharing_dt.describe()


#  Data Preparation for Linear Regression
#     Create dummy variables for all categorical variables

# In[65]:


bikeSharing_dt = pd.get_dummies(data=bikeSharing_dt,columns=["season","mnth","weekday"],drop_first=True)
bikeSharing_dt = pd.get_dummies(data=bikeSharing_dt,columns=["weathersit"])


# In[67]:


#Print columns after creating dummies
bikeSharing_dt.columns


# In[68]:


#Print few rows to inspect
bikeSharing_dt.head()


# Model Building
# Split Data into training and test

# In[69]:


#y to contain only target variable
y=bikeSharing_dt.pop('cnt')

#X is all remainign variable also our independent variables
X=bikeSharing_dt

#Train Test split with 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[70]:


#Inspect independent variables
X.head()


# In[71]:


# Checking shape and size for train and test
print(X_train.shape)
print(X_test.shape)


# Feature Scaling continuous variables
# To make all features in same scale to interpret easily
# 
# Following columns are continous to be scaled
# temp,hum,windspeed

# In[72]:


# Importing required library
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


# In[73]:


# Let us scale continuous variables
num_vars = ['temp','atemp','hum','windspeed','days_old']

#Use Normalized scaler to scale
scaler = MinMaxScaler()

#Fit and transform training set only
X_train[num_vars] = scaler.fit_transform(X_train[num_vars])


# In[74]:


#Inspect stats fro Training set after scaling
X_train.describe()


# In[75]:


X_train.head()


# Build a Model using RFE and Automated approach
# Use RFE to eliminate some columns

# In[76]:


# Build a Lienar Regression model using SKLearn for RFE
lr = LinearRegression()
lr.fit(X_train,y_train)


# In[78]:


#Cut down number of features to 15 using automated approach
rfe = RFE(lr)
rfe.fit(X_train,y_train)


# In[79]:


#Columns selected by RFE and their weights
list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# Manual elimination
# Function to build a model using statsmodel api

# In[80]:


#Function to build a model using statsmodel api - Takes the columns to be selected for model as a parameter
def build_model(cols):
    X_train_sm = sm.add_constant(X_train[cols])
    lm = sm.OLS(y_train, X_train_sm).fit()
    print(lm.summary())
    return lm


# In[81]:


#Function to calculate VIFs and print them -Takes the columns for which VIF to be calcualted as a parameter
def get_vif(cols):
    df1 = X_train[cols]
    vif = pd.DataFrame()
    vif['Features'] = df1.columns
    vif['VIF'] = [variance_inflation_factor(df1.values, i) for i in range(df1.shape[1])]
    vif['VIF'] = round(vif['VIF'],2)
    print(vif.sort_values(by='VIF',ascending=False))


# In[83]:


#Print Columns selected by RFE. We will start with these columns for manual elimination
X_train.columns[rfe.support_]


# In[84]:


# Features not selected by RFE
X_train.columns[~rfe.support_]


# In[86]:


# Taking 15 columns supported by RFE for regression
X_train_rfe = X_train[['yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jan', 'mnth_jul', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']]
X_train_rfe.shape


# Build Model
# Model 1 - Start with all variables selected by RFE

# In[87]:


#Selected columns for Model 1 - all columns selected by RFE
cols = ['yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jan', 'mnth_jul', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']

build_model(cols)
get_vif(cols)


# In[89]:


# Checking correlation of features selected by RFE with target column. 
# Also to check impact of different features on target.
plt.figure(figsize = (15,10))
sns.heatmap(bikeSharing_dt[['yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jan', 'mnth_jul', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']].corr(), cmap='GnBu', annot=True)
plt.show()


# In[90]:


# Model 2
# Dropping the variable mnth_jan as it has negative coefficient and is insignificant as it has high p-value
cols = ['yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jul', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# In[91]:


# Model 3
# All the columns have p-value > .05 so checking VIFs
# Dropping the variable hum as it has negative coefficient and is insignificant as it has high p-value
cols = ['yr', 'holiday', 'workingday', 'temp', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jul', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# In[93]:


# Model 4
# Dropping the variable holiday as it has negative coefficient and is insignificant as it has high p-value
cols = ['yr', 'workingday', 'temp', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jul', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# In[94]:


# Model 5
# Dropping the variable mnth_jul as it has negative coefficient and is insignificant as it has high p-value
cols = ['yr', 'workingday', 'temp', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# In[95]:


#Build a model with all columns to select features automatically
def build_model_sk(X,y):
    lr1 = LinearRegression()
    lr1.fit(X,y)
    return lr1


# In[96]:


#Let us build the finalmodel using sklearn
cols = ['yr', 'season_spring', 'mnth_jul',
        'season_winter', 'mnth_sept', 'weekday_sun',
       'weathersit_bad', 'weathersit_moderate', 'temp']

#Build a model with above columns
lr = build_model_sk(X_train[cols],y_train)
print(lr.intercept_,lr.coef_)


# Model Evaluation
# Residucal Analysis

# In[97]:


y_train_pred = lr.predict(X_train[cols])


# In[98]:


#Plot a histogram of the error terms
def plot_res_dist(act, pred):
    sns.distplot(act-pred)
    plt.title('Error Terms')
    plt.xlabel('Errors')


# In[99]:


plot_res_dist(y_train, y_train_pred)


# Errors are normally distribured here with mean 0. So everything seems to be fine
# 

# In[100]:


# Actual vs Predicted
c = [i for i in range(0,len(X_train),1)]
plt.plot(c,y_train, color="blue")
plt.plot(c,y_train_pred, color="red")
plt.suptitle('Actual vs Predicted', fontsize = 15)
plt.xlabel('Index')
plt.ylabel('Demands')
plt.show()


# In[101]:


#Actual and Predicted result following almost the same pattern so this model seems ok
# Error Terms
c = [i for i in range(0,len(X_train),1)]
plt.plot(c,y_train-y_train_pred)
plt.suptitle('Error Terms', fontsize = 15)
plt.xlabel('Index')
plt.ylabel('y_train-y_train_pred')
plt.show()


# In[102]:


# Here,If we see the error terms are independent of each other.
# Print R-squared Value
r2_score(y_train,y_train_pred)


# Observation:
#     R2 Same as we obtained for our final model

# In[103]:


# Linearity Check
# scatter plot for the check
residual = (y_train - y_train_pred)
plt.scatter(y_train,residual)
plt.ylabel("y_train")
plt.xlabel("Residual")
plt.show()


# In[104]:


# Predict values for test data set
#Scale variables in X_test
num_vars = ['temp','atemp','hum','windspeed','days_old']

#Test data to be transformed only, no fitting
X_test[num_vars] = scaler.transform(X_test[num_vars])
#Columns from our final model
cols = ['yr', 'season_spring', 'mnth_jul',
        'season_winter', 'mnth_sept', 'weekday_sun',
       'weathersit_bad', 'weathersit_moderate', 'temp']

#Predict the values for test data
y_test_pred = lr.predict(X_test[cols])


# In[105]:


# R-Squared value for test predictions
# Find out the R squared value between test and predicted test data sets.  
r2_score(y_test,y_test_pred)


# In[106]:


# Homoscedacity
# Inference
# R2 value for predictions on test data (0.815) is almost same as R2 value of train data(0.818). This is a good R-squared value, hence we can see our model is performing good even on unseen data (test data)

# Plotting y_test and y_test_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_test_pred)
fig.suptitle('y_test vs y_test_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_test_pred', fontsize = 16)


# Inference
# We can observe that variance of the residuals (error terms) is constant across predictions. 
# i.e., error term does not vary much as the value of the predictor variable changes.

# In[107]:


# Plot Test vs Predicted test values
#Function to plot Actual vs Predicted
#Takes Actual and PRedicted values as input along with the scale and Title to indicate which data
def plot_act_pred(act,pred,scale,dataname):
    c = [i for i in range(1,scale,1)]
    fig = plt.figure(figsize=(14,5))
    plt.plot(c,act, color="blue", linewidth=2.5, linestyle="-")
    plt.plot(c,pred, color="red",  linewidth=2.5, linestyle="-")
    fig.suptitle('Actual and Predicted - '+dataname, fontsize=20)              # Plot heading 
    plt.xlabel('Index', fontsize=18)                               # X-label
    plt.ylabel('Counts', fontsize=16)                               # Y-label


# In[108]:


#Plot Actual vs Predicted for Test Data
plot_act_pred(y_test,y_test_pred,len(y_test)+1,'Test Data')


# Inference
# As we can see predictions for test data is very close to actuals
# 
# Plot Error Terms for test data

# In[109]:


# Error terms
def plot_err_terms(act,pred):
    c = [i for i in range(1,220,1)]
    fig = plt.figure(figsize=(14,5))
    plt.plot(c,act-pred, color="blue", marker='o', linewidth=2.5, linestyle="")
    fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
    plt.xlabel('Index', fontsize=18)                      # X-label
    plt.ylabel('Counts - Predicted Counts', fontsize=16)                # Y-label
#Plot error terms for test data
plot_err_terms(y_test,y_test_pred)


# Inference
# As we can see the error terms are randomly distributed and there is no pattern which means 
# the output is explained well by the model and there are no other parameters that can explain the model better.

# Intrepretting the Model
# Let us go with interpretting the RFE with Manual model results as we give more importance to imputation
# 
# 

# In[110]:


#Let us rebuild the final model of manual + rfe approach using statsmodel to interpret it
cols = ['yr', 'season_spring', 'mnth_jul',
        'season_winter', 'mnth_sept', 'weekday_sun',
       'weathersit_bad', 'weathersit_moderate', 'temp']

lm = build_model(cols)


# Interepretation of results:
# 
# Analysing the above model, the comapany should focus on the following features:
# Company should focus on expanding business during Spring.
# Company should focus on expanding business during September.
# Based on previous data it is expected to have a boom in number of users once situation comes back to normal, compared to 2019.
# There would be less bookings during Light Snow or Rain, they could probably use this time to serive the bikes without having business impact.
# Hence when the situation comes back to normal, the company should come up with new offers during spring when the weather is pleasant and also advertise a little for September as this is when business would be at its best.
# 
# 
# Conclusion:
# 
# Significant variables to predict the demand for shared bikes
# 
# holiday
# temp
# hum
# windspeed
# Season
# months(January, July, September, November, December)
# Year (2019)
# Sunday
# weathersit( Light Snow, Mist + Cloudy)

# In[ ]:




