#!/usr/bin/env python
# coding: utf-8

# In[129]:


import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import random
import math
import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator
import matplotlib.pyplot as plt


# In[130]:


import sklearn
print(sklearn.__version__)


# # Membaca Data Kasus dan Jumlah Kematian akibat COVID di Indonesia Selama 10 Hari Terakhir (Data Disadur Dari Laporan European Centre for Disease Prevention and Control)

# In[131]:


df = pd.read_table(r'D:\IndonesiaCOVIDData.txt', sep=',')
df.tail(10)


# # Menyelipkan Angka Rasio ke Dalam Tabel

# In[132]:


healed = df['case'] - df['deaths']
healed_ratio = healed / df['case'] 
death_ratio = df['deaths']  / df['case'] 

if 'healed' in df.columns:
    df.drop('healed', axis=1, inplace=True)
if 'healed' not in df.columns:
    df.insert(loc=len(df.columns),column='healed',value = df['case'] - df['deaths'])
if 'healed_ratio' in df.columns:
    df.drop('healed_ratio', axis=1, inplace=True)
if 'healed_ratio' not in df.columns :
    df.insert(loc=len(df.columns),column='healed_ratio' ,value = healed / df['case']) 
if 'death_ratio' in df.columns:
    df.drop('death_ratio', axis=1, inplace=True)
if 'death_ratio' not in df.columns:
    df.insert(loc=len(df.columns),column='death_ratio',value = df['deaths']  / df['case'])      

df.tail(10)
    


# # Perbandingan Rasio Penyembuhan dan Kematian Akibat COVID di Indonesia Selama 10 Hari Terakhir

# In[133]:


healed = df['case'] - df['deaths']
healed_ratio = healed / df['case'] 
death_ratio = df['deaths']  / df['case'] 

if 'healed' in df.columns:
    df.drop('healed', axis=1, inplace=True)
if 'healed' not in df.columns:
    df.insert(loc=len(df.columns),column='healed',value = df['case'] - df['deaths'])
if 'healed_ratio' in df.columns:
    df.drop('healed_ratio', axis=1, inplace=True)
if 'healed_ratio' not in df.columns :
    df.insert(loc=len(df.columns),column='healed_ratio' ,value = healed / df['case']) 
if 'death_ratio' in df.columns:
    df.drop('death_ratio', axis=1, inplace=True)
if 'death_ratio' not in df.columns:
    df.insert(loc=len(df.columns),column='death_ratio',value = df['deaths']  / df['case'])   
    
plt.figure(figsize=(10,5))

plt.title('Rasio Penyembuhan dan Kematian akibat COVID di Indonesia', fontdict={'fontweight':'bold', 'fontsize': 18})

plt.plot(df.dates[78:87], df['healed_ratio'][78:87], 'g.-', label='Rasio Penyembuhan')
plt.plot(df.dates[78:87], df['death_ratio'][78:87], 'r.-', label='Rasio Kematian')

plt.xticks(df.dates[78:87])

plt.xlabel('Tanggal')
plt.ylabel('Rasio')

plt.legend()
plt.show()


# In[ ]:




