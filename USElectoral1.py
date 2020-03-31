#!/usr/bin/env python
# coding: utf-8

# In[170]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Membaca Data 

# In[171]:


txtfile = pd.read_table(r'C:\Users\ASUS-PC\Downloads\USAElectoral2016.txt', sep=',')


# In[172]:


txtfile


# # Mencari Tingkat Elektabilitas Ben Carson dan Jeb Bush di 50 Negara Bagian

# In[173]:


plt.figure(figsize=(30,10))

plt.title('Electability between Jeb Bush and Ben Carson', fontdict={'fontweight':'bold', 'fontsize': 18})

plt.plot(txtfile.State, txtfile.JebBush, 'b.-', label='JebBush')
plt.plot(txtfile.State, txtfile.BenCarson, 'r.-', label='BenCarson')

plt.xticks(txtfile.State)

plt.xlabel('State')
plt.ylabel('Electability')

plt.legend()
plt.show()


# # Mencari Elektabilitas Jeb Bush dan Ben Carson di 5 Negara Bagian Pertama

# In[174]:


plt.figure(figsize=(30,10))

plt.title('Electability between Jeb Bush and Ben Carson di DC,Delaware,Colorado,Maryland, dan Vermont', fontdict={'fontweight':'bold', 'fontsize': 18})

plt.plot(txtfile.State[0:5], txtfile.JebBush[0:5], 'b.-', label='JebBush')
plt.plot(txtfile.State[0:5], txtfile.BenCarson[0:5], 'r.-', label='BenCarson')

plt.xticks(txtfile.State[0:5])

plt.xlabel('State')
plt.ylabel('Electability')

plt.legend()
plt.show()


# # Mencari Elektabilitas Donald Trump di Swing State (Pennsylvania,New York,California,Florida)

# In[184]:



txtfile.iloc[[7,9,14,31], [19]].head()


# #  Mencari Elektabilitas Hillary Clinton di Swing State (Pennsylvania,New York,California,Florida)

# In[185]:


txtfile.iloc[[7,9,14,31], [5]].head()


# # Grafis Garis Elektabilitas Hillary Clinton dan Donald Trump di Swing State

# In[193]:


plt.figure(figsize=(30,10))

plt.title('Electability between Donald Trump and Hillary Clinton di Swing State', fontdict={'fontweight':'bold', 'fontsize': 18})

plt.plot(['Pennsylvania','NewYork','California','Florida'], ['9.67','12.72','9.45','8.44'], label='HillaryClinton')
plt.plot(['Pennsylvania','NewYork','California','Florida'],['60.84','73.22','78.30','73.05'], label='DonaldTrump')

plt.xticks(['Pennsylvania','NewYork','California','Florida'])

plt.xlabel('State')
plt.ylabel('Electability')

plt.legend()
plt.show()


# In[ ]:




