#!/usr/bin/env python
# coding: utf-8

# In[24]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


df = pd.read_csv(r"C:\Users\ASUS-PC\Downloads\Arrests.csv")
df.head(10)


# In[26]:


df.replace(to_replace=['Yes', 'No', 'White', 'Black', 'Male', 'Female'],value= ['1', '0', '1','2','1','2'], inplace=True)
df.head()


# In[27]:


plt.scatter(df.checks, df.age, color='blue')
plt.xlabel('Checks')
plt.ylabel('Age')
plt.show()


# In[31]:


cor=df.corr()

plt.figure(figsize=(25,25))
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.show()


# In[29]:


msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]


plt.scatter(train.checks, train.colour, color = 'blue')
plt.xlabel('checks')
plt.ylabel('colour')
plt.show()


# In[35]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['colour','sex','age','employed','citizen']])
y = np.asanyarray(train[['checks']])
regr.fit(x,y)

print('Coefficients: ', regr.coef_)


# In[36]:


y_= regr.predict(test[['colour','sex','age','employed','citizen']])
x = np.asanyarray(test[['colour','sex','age','employed','citizen']])
y = np.asanyarray(test[['checks']])
print('Residual sum squares: %.2f' % np.mean((y_ - y) **2 ))
print('Variance score: %.2f' % regr.score(x,y))


# In[38]:


regr = linear_model.LinearRegression()
x = np.asanyarray(train[['colour','sex','age','employed','citizen']])
y = np.asanyarray(train[['checks']])
regr.fit(x,y)

y_= regr.predict(test[['colour','sex','age','employed','citizen']])
x = np.asanyarray(test[['colour','sex','age','employed','citizen']])
y = np.asanyarray(test[['checks']])

print('Coefficients: ', regr.coef_)
print("Residual sum of squares: %.2f"% np.mean((y_ - y) ** 2))
print('Variance score: %.2f' % regr.score(x, y))


# In[41]:


reg = linear_model.LinearRegression()
reg.fit(df[['colour','sex','age','employed','citizen']],df.checks)


# In[43]:


reg.intercept_


# # Contoh Soal
# 
# Berapa kali seseorang yang berkulit hitam, berusia 25 tahun, seorang warga beridentitas, tidak bekerja, pria, akan diperiksa?

# In[53]:


0.46565268*2+-0.56008491*1+0.01930012*25+-0.79971643*0+0.12765156*1+1.7694251951865436

