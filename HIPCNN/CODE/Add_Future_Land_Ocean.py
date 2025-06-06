#!/usr/bin/env python
# coding: utf-8

# # Created by Sunny You on September 1, 2023
# 
# This script adds in the future of each storm's time of landfall to the model's training data.

# In[1]:


import numpy as np
import pandas as pd
import os
import re


# In[2]:


os.chdir(os.path.dirname(os.path.dirname(os.getcwd())))
print(os.getcwd())


# In[3]:


data = pd.read_csv('BRTK_SHIPS_2000to2019_IMERG_OK_Request_2023_FINAL.csv')
data['ID'] = data.GIS_ID.str[:10]
data = data[["GIS_ID", "JULDAY","LAND_OCEAN","ID"]]
data


# In[4]:


s = data[data.ID == "ATL_200001"]
s


# In[5]:


final_df = pd.DataFrame(columns = ["GIS_ID", "LOP06", "LOP12", "LOP18", "LOP24"])

i=0
for i in range(data.shape[0]):
    if i%1000 == 0:
        print(i)
    row = data.loc[i]
    cur_time = row.JULDAY
    temp = data[data.ID == row.ID]
    p6_time = cur_time + 0.25
    p12_time = cur_time + 0.5
    p18_time = cur_time + 0.75
    p24_time = cur_time + 1.0

    p6 = temp[temp.JULDAY == p6_time]
    p12 = temp[temp.JULDAY == p12_time]
    p18 = temp[temp.JULDAY == p18_time]
    p24 = temp[temp.JULDAY == p24_time]

    if p6.shape[0] == 0:
        p6 = "-999"
    else: 
        p6 = p6.LAND_OCEAN.iloc[0]

    if p12.shape[0] == 0:
        p12 = "-999"
    else: 
        p12 = p12.LAND_OCEAN.iloc[0]

    if p18.shape[0] == 0:
        p18 = "-999"
    else: 
        p18 = p18.LAND_OCEAN.iloc[0]

    if p24.shape[0] == 0:
        p24 = "-999"
    else: 
        p24 = p24.LAND_OCEAN.iloc[0]

    final_df.loc[i] = [row.GIS_ID, p6, p12, p18, p24]

final_df


# In[6]:


f = final_df
f["LOP12"] = np.where(f.LOP06 != "Ocean", f.LOP06, f.LOP12)
f["LOP18"] = np.where(f.LOP12 != "Ocean", f.LOP12, f.LOP18)
f["LOP24"] = np.where(f.LOP18 != "Ocean", f.LOP18, f.LOP24)


# In[7]:


f


# In[9]:


original_data = pd.read_csv('BRTK_SHIPS_2000to2019_IMERG_OK_Request_2023_FINAL.csv')
original_data


# In[10]:


original_data = original_data.merge(f, on = "GIS_ID")
original_data


# In[11]:


original_data.to_csv("HPCNN/IMERG/Land_Ocean_Futures.csv")

