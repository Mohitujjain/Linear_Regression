#!/usr/bin/env python
# coding: utf-8

# <div style="text-align: center;"><div  style="color:#7f0000; font-size:30px; font-weight:bold; line-height:40px;">Multiple Linear Regression (Sales Price Prediction)</div></div>
# <div style="text-align: center; color:#006666"><strong>Owner: </strong>Mohit Kumar</div>
# <div style="text-align: center; color:#006666"><strong>Mail ID: </strong>mohitujjain71195@gmail.com</div>
# <div style="text-align: center; color:#006666"><strong>Linkedin ID: </strong>https://www.linkedin.com/in/mohit-kumar-61bb20198</div>
# <div style="text-align: center; color:#006666"><strong>Github ID: </strong> https://github.com/Mohitujjain</div>                                                                                                                                 

# # Objective
# 
#   * Identify the Which Platform have more impact on Sales
#   
#   * To build the Model which will help to Predict the future sales based on Money invest in Different Platform

# In[1]:


import pandas as pd
import numpy as np
import statsmodels
from statsmodels.stats.anova import anova_lm
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.stats.outliers_influence import variance_inflation_factor


# # Data Import

# In[3]:


new = pd.read_csv(r"E:\Data Science\Irfaan Lec\Statistics\ProbabilityDist\Linear Regression\Advertising_Data.csv")


# In[4]:


new.head()


# In[5]:


new.tail()


# In[6]:


# Let's Look at some statistical information about our dataframe.

new.describe()


# In[7]:


sns.boxplot( data=new)


# # Winsorizing Technique -- Treatment of Outlier

# In[8]:


q1 = new['newspaper'].quantile(0.25)
q3 = new['newspaper'].quantile(0.75)
iqr = q3-q1 #Interquartile range
low_limit = q1-1.5*iqr #acceptable range
upper_limit = q3+1.5*iqr #acceptable range
low_limit,upper_limit


# In[9]:


q1,q3,iqr


# In[10]:


new['newspaper']=np.where(new['newspaper'] > upper_limit,upper_limit,new['newspaper']) # upper limit


# In[11]:


sns.boxplot(data=new)


# # Step 3 : Splitting the data in Training and Test set
#     
#     * Using sklearn we split 70% of our data into training set and rest in test set.
#     
#     * Setting random_state will give the same training and test set everytime on running the code

# In[13]:


# Putting feature variable to X
X = new.drop('sales',axis=1)

# Putting response variable to y
y = new['sales']

#random_state is the seed used by the random number generator. It can be any integer.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=141)


# In[14]:


df = pd.concat([y_train, X_train], axis=1)
df.head()


# # Correlation Matrix

# In[15]:


df.corr()


# In[16]:


sns.heatmap(df.corr(),annot = True)


# In[17]:


df.plot.scatter(x='TV', y='sales', title='sales vs TV')


# # TV have 76% strong positive correlation with Sales
# 
#    * TV have high impact on Sales

# # Multicolinearity (with the help of VIF)

# In[18]:


X1=df.drop(['sales'],axis=1)
series_before = pd.Series([variance_inflation_factor(X1.values, i)
                          for i in range(X1.shape[1])],
                         index=X1.columns)
series_before


# ### No multi-collinearity in the data

# In[23]:


#X1.columns


# In[24]:


#X1.shape[1]


# # Model Building
# 
# 
# ### Hypothesis Testing
# 
# 
#    * HO :- There is no linear relationship between sales and tv, radio & newspaper
#    
#    * Vs
#    
#    * H1 :- There is linear relationship between sales and tv, radio & newspaper
#    
#  ### Alpha  = 5%(0.05)

# In[25]:


import statsmodels.formula.api as smf
model = smf.ols('sales ~ TV+ radio+newspaper', data=df).fit()
model.summary()


# #### There is  linear relationship between sales and tv, radio .
# 
# #### There is no  linear relationship between sales and newspaper.
# 
#    #### So we will remove and re-run the model
#    
# #### 88% is accuracy of model, so we say model is good fit

# In[26]:


# removing the insignificant variable
model2=smf.ols('sales ~ TV + radio', data=df).fit()
model2.summary()


# #### There is Significane relationship between sales and tv, radio
# 
# #### 88% is acccuracy of model, so we say model is good fit
# 
# #### Y = 2.9474 + 0.0466*TV + 0.1811*Radio

# In[27]:


X1=df.loc[:,['TV', 'radio']]
series_before = pd.Series([variance_inflation_factor(X1.values, i)
                          for i in range(X1.shape[1])], index=X1.columns)

series_before


# # Assumption of model
# 
#    * Linearity
#    * Homoscedasicity
#    * Normality
#    * Model Error has to be independently identically Distributed

# In[28]:


Y = 2.9474 + 0.0466 * 241.7 + 0.1811 * 38
Y


# In[29]:


df['fitted_value']=model2.fittedvalues #predicted value
df['residual']=model2.resid  #Error
df.head()


# # Linearity

# In[32]:


df.plot.scatter(x='TV', y='sales', title='sales vs TV')


# # Homoscedasicty

# In[33]:


#p = df.plot.scatter(x='fitted_value', y='residual')
#plt.xlabel('Fitted values')
#plt.ylabel('Residuals')
#p = plt.title('Residuals vs fitted values plot for homoscedasricity check')
#plt.show()


# In[35]:


sns.scatterplot(x='fitted_value', y='residual',data=df)


# # Normality

# In[37]:


import statsmodels.api as sm
fig = sm.qqplot(df['residual'], fit=True, line='s')

# s indicate standardized line
plt.show()


# # Model Error are IID

# In[38]:


df['residual'].plot.hist()


# In[39]:


ax = sns.distplot(df.residual)


# # Prediction on Test Data (unseen data)

# In[40]:


df_test=pd.concat([X_test,y_test], axis=1)


# In[41]:


df_test.head()


# In[42]:


df_test['Prediction']=model2.predict(df_test)
df_test.head() 


# In[43]:


Y = 2.9474 + 0.0466 * 110.7 + 0.1863 * 40.6
Y


# # Finish

# # MSE & MAE & RMSE for evaluation of Model on train data

# In[48]:


import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# calculate the mean squared error
model_mse = mean_squared_error(df['sales'],df['fitted_value'])
# calculate the root mean squared error
model_rmse = math.sqrt(model_mse)

print("RMSE {:.3}".format(model_rmse))


# # MSE & MAE & RMSE for evaluation of Model on test data

# In[49]:


import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# calculate the mean squared error
model_mse = mean_squared_error(df['sales'],df['Prediction'])
# calculate the root mean squared error
model_rmse = math.sqrt(model_mse)

print("RMSE {:.3}".format(model_rmse))


# In[ ]:




