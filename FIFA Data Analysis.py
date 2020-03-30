#!/usr/bin/env python
# coding: utf-8

# In[86]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# In[124]:


df = pd.read_csv(r"C:\Users\ASUS-PC\Downloads\matplotlib_tutorial-master\fifa_data.csv") 
df.columns = [x.strip().replace(' ', '_') for x in df.columns]


# In[129]:


df.head(10)


# In[113]:


df.shape


# In[136]:


df.describe()


# # Mengetahui Persebaran Skill dari Seluruh Pemain di FIFA 2018

# In[4]:


bins = [40,50,60,70,80,90,100]

plt.figure(figsize=(8,5))

plt.hist(df.Overall, bins=bins, color='#abcdef')

plt.xticks(bins)

plt.ylabel('Number of Players')
plt.xlabel('Skill Level')
plt.title('Distribution of Player Skills in FIFA 2018')

plt.savefig('histogram.png', dpi=300)

plt.show()


# # Mengetahui Distribusi Umur dari Seluruh Pemain di FIFA 2018
# 

# In[7]:


bins = [15,20,30,40]

plt.figure(figsize=(8,5))

plt.hist(df.Age, bins=bins, color='#1cccd9')

plt.xticks(bins)

plt.ylabel('Number of Players')
plt.xlabel('Age')
plt.title('Distribution of Player Age in FIFA 2018')

plt.savefig('histogram.png', dpi=300)

plt.show()


# # Mengetahui Persebaran Gaji dari Seluruh Pemain FIFA 2018

# In[55]:


plt.figure(figsize=(8,5), dpi=100)

plt.style.use('ggplot')

df.Wage = [int(x.strip('€K')) if type(x)==str else x for x in df.Wage]

Too_Low = df.loc[df.Wage < 100].count()[0]
Low = df[(df.Wage >= 100) & (df.Wage < 200)].count()[0]
Average = df[(df.Wage >= 200) & (df.Wage < 300)].count()[0]
Above_Average = df[(df.Wage >= 300) & (df.Wage < 400)].count()[0]
Extraordinary = df[(df.Wage >= 400) & (df.Wage < 500)].count()[0]
Superb = df[df.Wage >= 500].count()[0]

Wage = [Too_Low,Low, Average, Above_Average, Extraordinary, Superb]
labels = ['under 100', '100-200', '200-300', '300-400', '400-500','over 500']
explode = (0,1,2,3,4,5)

plt.title('Distribution of FIFA Player Wage')

plt.pie(Wage, labels=labels, explode=explode, pctdistance=0.8,autopct='%.2f %%')
plt.show()


# In[56]:


print (df.Nationality.unique())


# # Mengetahui Persentase Atlet FIFA 2018 Dari Spanyol, Argentina, Portugal, Belgia, dan Brazil

# In[57]:


Spain = df.loc[df['Nationality'] == 'Spain'].count()[0]
Argentina = df.loc[df['Nationality'] == 'Argentina'].count()[0]
Portugal = df.loc[df['Nationality'] == 'Portugal'].count()[0]
Belgium = df.loc[df['Nationality'] == 'Belgium'].count()[0]
Brazil = df.loc[df['Nationality'] == 'Brazil'].count()[0]

plt.figure(figsize=(8,5))

labels = ['Spain', 'Argentina','Portugal','Belgium','Brazil']
colors = ['#fcba03', '#aabbcc','#4aad34','#cf7686','#3c15e8' ]

plt.pie([Spain,Argentina,Portugal,Belgium,Brazil], labels = labels, colors=colors, autopct='%.2f %%')

plt.title('Player Distribution in FIFA 2018 from Spain, Argentina, Portugal, Belgium and Brazil')

plt.show()


# # Mengetahui Persentase Atlet Yang Memakai Kaki Kiri dan Kaki Kanan

# In[126]:


left = df.loc[df['Preferred_Foot'] == 'Left'].count()[0]
right = df.loc[df['Preferred_Foot'] == 'Right'].count()[0]

plt.figure(figsize=(8,5))

labels = ['Left', 'Right']
colors = ['#abcdef', '#aabbcc']

plt.pie([left, right], labels = labels, colors=colors, autopct='%.2f %%')

plt.title('Foot Preference of FIFA Players')

plt.show()


# In[93]:


print(df.columns)


# In[108]:


df.columns = [x.strip().replace(' ', '_') for x in df.columns]


# # Mengetahui Korelasi Antara Stamina dan Reputasi Internasional di FIFA 2018

# In[122]:


plt.figure(figsize=(12,5))
sns.scatterplot(x='Overall',y='Stamina',hue='International_Reputation',data=df)
plt.show()

