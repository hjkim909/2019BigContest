#!/usr/bin/env python
# coding: utf-8

# # 라이온킹 모델링 코드

# In[1]:


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import pickle


# # MODEL A : 분류 후 회귀

# In[2]:


train = pd.read_csv('../preprocess/train_preprocess_1.csv')


# In[3]:


train.shape


# In[4]:


def retained(x):
    if x==64:
        return 1
    else:
        return 0


# In[5]:


train['retained'] = train['survival_time'].apply(retained)


# In[6]:


train1 = train[train['최초접속일']==1]
train2 = train[train['최초접속일']==28]
train3 = train[~train['최초접속일'].isin([1,28])]


# ### Survival Time

# #### 2-1. GROUP 1
# ##### 분류

# In[7]:


# 분류
X_train1분류 = train1.iloc[:, 1:-3]
y_train1분류 = train1['retained']


# In[8]:


model1_group1_survival_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=50).fit(X_train1분류, y_train1분류)


# In[9]:


# Save Model
filename = 'final_model_1.sav'
pickle.dump(model1_group1_survival_clf, open(filename, 'wb'))


# ##### 회귀

# In[10]:


# train1 이탈자의 데이터로만 트레이닝 
이탈자 = train1[train1['retained']==0]


# In[11]:


X_train1회귀 = 이탈자.iloc[:, 1:-3]
y_train1회귀 = 이탈자['survival_time']


# In[12]:


model1_group1_survival_reg = lgb.LGBMRegressor(objective='quantile',
                        alpha= 0.09,
                        num_leaves=5,
                        learning_rate=0.1, n_estimators=1100,
                        num_iterations=3000,
                        max_bin = 500, bagging_fraction = 1,
                        bagging_freq = 20, feature_fraction = 0.2319,
                        feature_fraction_seed=9, bagging_seed=42,
                        min_data_in_leaf =10, min_sum_hessian_in_leaf = 15, random_state=42).fit(X_train1회귀, y_train1회귀)


# In[13]:


# Save Model
filename = 'final_model_2.sav'
pickle.dump(model1_group1_survival_reg, open(filename, 'wb'))


# ### 2-2. Group 2

# In[14]:


이탈자 = train2[train2['retained']==0]


# In[15]:


X_train2 = 이탈자.iloc[:, 1:-3]
y_train2 = 이탈자['survival_time']


# In[16]:


model1_group2_survival = lgb.LGBMRegressor(objective='quantile',
                        alpha= 0.09,
                        num_leaves=5,
                        learning_rate=0.1, n_estimators=1100,
                        num_iterations=3000,
                        max_bin = 500, bagging_fraction = 1,
                        bagging_freq = 20, feature_fraction = 0.2319,
                        feature_fraction_seed=9, bagging_seed=42,
                        min_data_in_leaf =10, min_sum_hessian_in_leaf = 15, random_state=42).fit(X_train2, y_train2)


# In[17]:


# Save Model
filename = 'final_model_3.sav'
pickle.dump(model1_group2_survival, open(filename, 'wb'))


# ### 3-3. Group 3

# In[18]:


이탈자 = train3[train3['retained']==0]


# In[19]:


X_train3 = 이탈자.iloc[:, 1:-3]
y_train3 = 이탈자['survival_time']


# In[20]:


model1_group3_survival =lgb.LGBMRegressor(objective='quantile',
                        alpha= 0.09,
                        num_leaves=5,
                        learning_rate=0.1, n_estimators=1100,
                        num_iterations=3000,
                        max_bin = 500, bagging_fraction = 1,
                        bagging_freq = 20, feature_fraction = 0.2319,
                        feature_fraction_seed=9, bagging_seed=42,
                        min_data_in_leaf =10, min_sum_hessian_in_leaf = 15, random_state=42).fit(X_train3, y_train3)


# In[21]:


# Save Model
filename = 'final_model_4.sav'
pickle.dump(model1_group3_survival, open(filename, 'wb'))


# ### amount_spent

# In[22]:


train = train[train['amount_spent'] != 0]
train = train[train['survival_time'] != 64]

X = train.iloc[:,1:-3]
y = train['amount_spent']


# In[23]:


model1_amount = lgb.LGBMRegressor(objective='quantile',
                        alpha= 0.98,
                        num_leaves=5,
                        learning_rate=0.1, n_estimators=1100,
                        num_iterations=3000,
                        max_bin = 500, bagging_fraction = 1,
                        bagging_freq = 20, feature_fraction = 0.2319,
                        feature_fraction_seed=9, bagging_seed=42,
                        min_data_in_leaf =10, min_sum_hessian_in_leaf = 15, random_state=42).fit(X, y)


# In[24]:


# Save Model
filename = 'final_model_5.sav'
pickle.dump(model1_amount, open(filename, 'wb'))


# In[ ]:





# # MODEL B : 바로 회귀

# In[25]:


train = pd.read_csv('../preprocess/train_preprocess_2.csv')


# ### survival_time

# In[26]:


train = train[train['amount_spent'] != 0]
train = train[train['survival_time'] != 64]


# In[27]:


X = train.iloc[:,1:-2]
y = train.iloc[:, -2]


# In[28]:


model2_survival = lgb.LGBMRegressor(objective='quantile',
                        alpha= 0.07,
                        num_leaves=10,
                        learning_rate=0.01, n_estimators=1000,
                        num_iterations=3000,
                        max_bin = 500, bagging_fraction = 1,
                        bagging_freq = 20, feature_fraction = 0.2319,
                        feature_fraction_seed=9, bagging_seed=42,
                        min_data_in_leaf =10, min_sum_hessian_in_leaf = 15, random_state=42).fit(X,y)


# In[29]:


# Save Model
filename = 'final_model_6.sav'
pickle.dump(model2_survival, open(filename, 'wb'))


# ### amount_spent

# In[30]:


X = train.iloc[:,1:-2]
y = train.iloc[:, -1]


# In[31]:


model2_amount = lgb.LGBMRegressor(objective='quantile',
                        alpha= 0.98,
                        num_leaves=10,
                        learning_rate=0.01, n_estimators=1000,
                        num_iterations=3000,
                        max_bin = 500, bagging_fraction = 1,
                        bagging_freq = 20, feature_fraction = 0.2319,
                        feature_fraction_seed=9, bagging_seed=42,
                        min_data_in_leaf =10, min_sum_hessian_in_leaf = 15, random_state=42).fit(X,y)


# In[32]:


# Save Model
filename = 'final_model_7.sav'
pickle.dump(model2_amount, open(filename, 'wb'))

