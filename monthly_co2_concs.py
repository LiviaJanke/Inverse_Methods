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

year,month,decimal_date,average,deseasonalized,ndays,sdev, unc = np.loadtxt('Data/monthly_co2_conc.csv', skiprows = 53, delimiter = ',', unpack = True)


monthly_co2_df = pd.read_csv('Data/monthly_co2_conc.csv', skiprows = 52)

monthly_temp_df = pd.read_csv('Data/temp_monthly.csv')


df_gistemp = monthly_temp_df[monthly_temp_df.Source != "GCAG"]


df_gistemp['Date'] = pd.to_datetime(df_gistemp['Date'])

#%%


#plotting temperature

df_gistemp.plot('Date','Mean', title = 'monthly mean temperature anomaly (C)')

#plotting co2

#%%

monthly_co2_df.plot('decimal_date','average', title = 'monthly mean co2 concentration (ppm)')






























