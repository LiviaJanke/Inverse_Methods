# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 21:02:02 2022

@author: Admin
"""
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#%%

#year,month,decimal_date,average,deseasonalized,ndays,sdev,unc = np.loadtxt('Data/monthly_co2_conc.csv', skiprows = 53, delimiter = ',', unpack = True)
year,month,decimal_date,average,deseasonalized,ndays,sdev,unc = np.loadtxt('Data/co2_month.csv', skiprows = 53, delimiter = ',', unpack = True)
#monthly_co2_df = pd.read_csv('Data/monthly_co2_conc.csv', skiprows = 52)

monthly_temp_df = pd.read_csv('Data/temp_monthly.csv')
monthly_co2_df = pd.read_csv('Data/co2_month.csv', skiprows = 52)

df_gistemp = monthly_temp_df[monthly_temp_df.Source != "GCAG"]


df_gistemp['Date'] = pd.to_datetime(df_gistemp['Date'])

#%%


#plotting temperature

df_gistemp.plot('Date','Mean', title = 'monthly mean temperature anomaly (C)')

#plotting co2

#%%

monthly_co2_df.plot('decimal_date',['average', 'deseasonalized'],title = 'monthly mean co2 concentration (ppm)')
#monthly_co2_df.plot('decimal_date','deseasonalized', title = 'monthly mean co2 concentration (ppm)')

#%%


def forward_poly(t,n):
    
    Ki = np.array([t**j for j in range (n)])
    
    return Ki

#%%

n = 5

m = len(year)

y = average

K = np.zeros([m,n])

for i in range(m):
    K[i,:] = forward_poly(year[i],n)
    
    
print(np.linalg.matrix_rank(K))

print(K)

#%%

xlsq, res, rank, svd = np.linalg.lstsq(K,y,rcond = -1)

ylsq = K.dot(xlsq)

residuals = y-ylsq




#%%
#plt.plot(year, deseasonalized)
plt.plot(year, ylsq, label = 'least squares soln')
plt.plot(year, average, label = 'co2')
plt.show()


#%%

plt.plot(year, residuals)
plt.show()

#%%



























