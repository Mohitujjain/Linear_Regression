#!/usr/bin/env python
# coding: utf-8

# <div style="text-align: center;"><div  style="color:#7f0000; font-size:30px; font-weight:bold; line-height:40px;">Simple Linear Regression</div></div>
# <div style="text-align: center; color:#006666"><strong>Owner: </strong>Mohit kumar</div>
# <div style="text-align: center; color:#006666"><strong>Mail ID: </strong>mohitujjain@gmail.com</div>
# <div style="text-align: center; color:#006666"><strong>Linkedin ID: </strong>https://https://www.linkedin.com/in/mohit-kumar-61bb20198/</div>

# # Business Problem
# 
# #### Apply the simple linear regression model for the data set Salary.
# #### Decide whether there is a significant relationship between the variables in the linear regression model of the data set Salary at 5% significance level.

# In[3]:


import numpy as np #Data Calculation
import pandas as pd #Data frame



import matplotlib.pyplot as plt  #Data visualization
import seaborn as sns ##Data visualization

from scipy import stats
import statsmodels.api as sm

from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')
import os 


# # Data Import

# In[5]:


# Industry way - Same step, sligthly different 

filepath = r"E:\Data Science\Irfaan Lec\Statistics\ProbabilityDist\Linear Regression"
filename = "Salary_Data.csv"

new = pd.read_csv(os.path.join(filepath, filename))


# In[6]:


# Looking at the first five rows
new.head()


# In[7]:


# Looking at the last five rows
new.tail()


# # Data Type

# In[8]:


# What type of values are stored in the columns?
new.info()


# # Univariate Analysis

# In[9]:


# Let's Look at some statistical information about
# our dataframe
new.describe()


# # Identify & Treatment of Outlier

# In[10]:


sns.boxplot(y='YearsExperience', data=new)


# In[11]:


sns.boxplot(y='Salary', data=new)


# # Missing Value

# In[12]:


x=new[['YearsExperience']]
y=new[['Salary']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x,y,train_size=0.7,random_state=131)


# In[13]:


train = pd.concat([X_train,y_train], axis=1)

train.head()


# In[14]:


np.mean(train.Salary)


# # Bi-variate Analysis

# In[15]:


sns.scatterplot(x='YearsExperience', y='Salary',data=train)


# In[16]:


train.corr()


# In[17]:


sns.heatmap(train.corr(),annot = True)


# # Model Building
# 
# 
# ### Hypothesis Testing
# 
# 
#    * HO :- There is no relationship between the Salary & YearsExperience
#    
#    * Vs
#    
#    * H1 :- There is  relationship between the Salary & YearsExperience
#    
#  ### Alpha  = 5%(0.05)

# In[19]:


import statsmodels.formula.api as smf
reg = smf.ols('Salary ~ YearsExperience', data=train).fit()
reg.summary()


# # Conclusion
# 
#   * There is relationship between the Salary & YearsExperience
#   
#   
# # Assumption of Model
#    
#    * Linearity
#    * Homoscedasicity
#    * Normality
#    * Model Error

# In[22]:


sns.scatterplot(x='Salary', y='YearsExperience',data=train)


# In[23]:


train['fitted_value']=reg.fittedvalues
train['residual']=reg.resid

train.head()


# In[24]:


y=25.360+9.5641*5.3
y


# # Homoscedasicty

# In[25]:


sns.scatterplot(x='fitted_value', y='residual', data=train)


# # Normality

# In[26]:


fig = sm.qqplot(train['residual'], fit=True, line='s')

# s indicate standardized line
plt.show()


# # Model Error are ID

# In[27]:


ax = sns.distplot(train.residual)


# # Prediction on Test Data (unseen data)

# In[28]:


test=pd.concat([X_test,y_test], axis=1)
test.head()


# In[29]:


y=25.360+9.5641*3.2
y


# In[30]:


test['Predicted']=reg.predict(test)
test.head() 


# # Finish

# # MSE & MAE & RMSE for evaluation of Model on train data

# In[32]:


import math
# calculate the mean squared error
model_mse = mean_squared_error(train['Salary'],train['fitted_value'])
# calculate the root mean squared error
model_rmse = math.sqrt(model_mse)

print("RMSE {:.3}".format(model_rmse))


# # MSE & MAE & RMSE for evaluation of Model on test data

# In[34]:


import math
# calculate the mean squared error
model_mse = mean_squared_error(test['Salary'],test['Predicted'])
# calculate the root mean squared error
model_rmse = math.sqrt(model_mse)

print("RMSE {:.3}".format(model_rmse))


# In[ ]:




