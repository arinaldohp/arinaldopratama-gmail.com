#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 10,6


# In[4]:


df = pd.read_csv("D:\international-airline-passengers.csv")
df['Month'] = pd.to_datetime(df['Month'], infer_datetime_format=True)
indexed = df.set_index(['Month'])


# In[107]:


from datetime import datetime
indexed.head(10)


# In[108]:


df.columns = ["month", "monthly_totals"]
df.head()


# In[109]:


plt.xlabel('Month')
plt.ylabel('monthly_totals')
plt.plot(indexed)


# In[110]:


rolmean = indexed.rolling(window=12).mean()
rolstd = indexed.rolling(window=12).std()
print(rolmean,rolstd)


# In[111]:


orig = plt.plot(indexed, color = 'red', label = 'original')
mean = plt.plot(rolmean, color='blue', label='mean')
std = plt.plot(rolstd, color='black', label='std')
plt.legend(loc='best')
plt.title('Rolling Mean and Rolling Standard Deviation')
plt.show(block=False)


# In[112]:


from statsmodels.tsa.stattools import adfuller

print ('Result of Dickey-Fuller Test')
dftest = adfuller(indexed['International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60'])

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Numer of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
    
print(dfoutput)


# In[113]:


IndexedDataSet_logScale = np.log(indexed)
plt.plot(indexed)


# In[114]:


movingAverage = np.log(indexed).rolling(window=12).mean()
movingSTD =  np.log(indexed).rolling(window=12).std()
plt.plot(IndexedDataSet_logScale)
plt.plot(movingAverage, color='purple')


# In[115]:


datasetLogScaleMinusMovingAverage = IndexedDataSet_logScale - movingAverage
datasetLogScaleMinusMovingAverage.head()

datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(12)


# In[116]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    movingAverage = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()
    
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(movingAverage, color='red',label='Mean')
    std = plt.plot(movingSTD,color='black',label='STD')
    plt.legend(loc='best')
    
    
    print('Result of Dickey Fuller Test')
    dftest = adfuller(indexed['International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60'])
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Numer of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)

    
    


# In[117]:


test_stationarity(datasetLogScaleMinusMovingAverage)


# In[118]:


exponentialdecayweightedaverage = IndexedDataSet_logScale.ewm(halflife=12, min_periods=8, adjust=True).mean()
plt.plot(IndexedDataSet_logScale)
plt.plot(exponentialdecayweightedaverage, color='red')


# In[119]:


datasetLogScaleMinusMovingExponentDecayAverage = IndexedDataSet_logScale - exponentialdecayweightedaverage
test_stationarity(datasetLogScaleMinusMovingExponentDecayAverage)


# In[120]:


datasetLogDiffShifting = IndexedDataSet_logScale - IndexedDataSet_logScale.shift()
test_stationarity(datasetLogDiffShifting)


# In[121]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(IndexedDataSet_logScale)

trend =decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(IndexedDataSet_logScale, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residual')
plt.legend(loc='best')

decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)


# In[122]:


decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)


# In[123]:


from statsmodels.tsa.stattools import acf, pacf



# In[127]:


lag_acf = acf(datasetLogDiffShifting, nlags=20)
lag_pacf = pacf(datasetLogDiffShifting, nlags=20)


# In[142]:


from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(IndexedDataSet_logScale, order=(2,1,2))
results_AR = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-datasetLogDiffShifting['International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60'])**2))
print('Plotting AR Model')


# In[146]:


model = ARIMA(IndexedDataSet_logScale, order=(0,1,2))
results_ARIMA = model.fit(disp=1)
predictions_Arima_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_Arima_diff.head(20))


# In[147]:


prediction_ARIMA_cumsum = predictions_Arima_diff.cumsum()
print(prediction_ARIMA_cumsum)


# In[149]:


prediction_ARIMA_log = pd.Series(IndexedDataSet_logScale['International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60'].ix[0], index=IndexedDataSet_logScale.index)
prediction_ARIMA_log = prediction_ARIMA_log.add(prediction_ARIMA_cumsum,fill_value=0)
prediction_ARIMA_log.head(10)


# In[150]:


prediction_ARIMA = np.exp(prediction_ARIMA_log)
plt.plot(indexed)
plt.plot(prediction_ARIMA)


# In[151]:


IndexedDataSet_logScale


# In[154]:


results_ARIMA.plot_predict(1,155)


# In[ ]:




