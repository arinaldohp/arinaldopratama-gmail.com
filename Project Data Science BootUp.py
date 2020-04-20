#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv(r"C:\Users\ASUS-PC\Desktop\maintenance_data.csv") 
df.head(30)


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


df.shape


# In[15]:


print (df.moistureInd.unique())


# In[16]:



print (df.team.unique())


# In[17]:



print (df.provider.unique())


# In[27]:


for col in ["broken","provider","team"]:
    plt.figure(figsize=(6,3))
    sns.countplot(df[col])
    plt.show()


# In[25]:


for col in ["lifetime"]:
    plt.figure(figsize=(50,40))
    sns.countplot(df[col])
    plt.show()


# In[28]:


out = pd.crosstab(df["team"],df["broken"], margins = True)
out


# In[29]:


out = pd.crosstab(df["provider"],df["broken"], margins = True)
out


# In[34]:


plt.figure(figsize=(12,5))
sns.distplot(df.lifetime[df.broken==0])
sns.distplot(df.lifetime[df.broken==1])
plt.legend(['0','1']) 
plt.show()


# In[44]:


plt.figure(figsize=(12,5))
sns.distplot(df.moistureInd[df.broken==0])
sns.distplot(df.moistureInd[df.broken==1])
plt.legend(['0','1']) 
plt.show()


# In[47]:


df.isnull().sum()


# In[55]:


plt.figure(figsize=(12,5))
sns.scatterplot(x='moistureInd',y='temperatureInd',hue='broken',data=df)
plt.show()


# In[56]:


plt.figure(figsize=(12,5))
sns.scatterplot(x='pressureInd',y='temperatureInd',hue='broken',data=df)
plt.show()


# In[70]:


for col in ["lifetime","temperatureInd","pressureInd","moistureInd"]:
    plt.figure(figsize=(8,4))
    plt.scatter(np.arange(1000),df[col])
    plt.title(col)
    plt.show()


# In[88]:


cor=df.corr()

plt.figure(figsize=(25,25))
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.show()


# In[81]:


plt.figure(figsize=(10,9))
sns.scatterplot(x='provider',y='lifetime',hue='broken',data=df)
plt.show()


# In[ ]:




