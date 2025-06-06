#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author: Sunny You
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras


# In[ ]:


os.chdir("/Users/sunnyyou/Documents/Real_Time_HAI/HRICNN")


# In[2]:


train = pd.read_csv("IMERG/Model_Data/ATL_train_resample.csv")
train = train[["GIS_ID", "DATE", "VMAX", "SHIPS_PER", "SHIPS_POT_Avg24h", "SHIPS_NOHC_Avg24h", "SHIPS_SHDC_Avg24h", "SHIPS_CFLX_Avg24h", "SHIPS_D200_Avg24h", "SHIPS_MTPW_108h", "SHIPS_PC2", "SHIPS_IR00_12h", "Category", "RI"]]
train.set_axis(["GIS_ID", "DATE", "VMAX", "PER", "POT", "NOHC", "SHDC", "ICDA", "D200", "TPW", "PC2", "SDBT", "Category", "RI"], axis = 1, inplace = True)
train


# In[3]:


test = pd.read_csv("IMERG/Model_Data/ATL_2018_test.csv")
test = test[["GIS_ID", "DATE", "VMAX", "SHIPS_PER", "SHIPS_POT_Avg24h", "SHIPS_NOHC_Avg24h", "SHIPS_SHDC_Avg24h", "SHIPS_CFLX_Avg24h", "SHIPS_D200_Avg24h", "SHIPS_MTPW_108h", "SHIPS_PC2", "SHIPS_IR00_12h", "Category", "RI"]]
test.set_axis(["GIS_ID", "DATE", "VMAX", "PER", "POT", "NOHC", "SHDC", "ICDA", "D200", "TPW", "PC2", "SDBT", "Category", "RI"], axis = 1, inplace = True)
test


# In[4]:


train_img = []
train_ships = []
train_label = []
test_img = []
test_ships = []
test_label = []
for f in range(len(train.GIS_ID)) :
    filename = "IMERG_Data_Old/IMERG_CSV/" + train.GIS_ID[f] + ".csv"
    try:
        temp = pd.read_csv(filename, header = None)
        if (temp.shape != (121,121)):
            continue
        temp = temp[30:91, 30:91]
        temp = np.array(temp)
        train_img.append(temp)
        lab = train.RI[f]
        train_label.append(lab)
        ships = np.array([train.VMAX[f], train.PER[f], train.POT[f], train.NOHC[f], train.SHDC[f], train.ICDA[f], train.D200[f], train.TPW[f], train.PC2[f], train.SDBT[f]])
        train_ships.append(ships)
    except Exception as e:
        pass

for f in range(len(test.GIS_ID)) :
    filename = "IMERG_Data_Old/IMERG_CSV/" + test.GIS_ID[f] + ".csv"
    try:
        temp = pd.read_csv(filename, header = None)
        if (temp.shape != (121,121)):
            continue
        temp = temp.iloc[30:91, 30:91]
        temp = np.array(temp)
        test_img.append(temp)
        lab = test.RI[f]
        test_label.append(lab)
        ships = np.array([test.VMAX[f], test.PER[f], test.POT[f], test.NOHC[f], test.SHDC[f], test.ICDA[f], test.D200[f], test.TPW[f], test.PC2[f], test.SDBT[f]])
        test_ships.append(ships)
    except Exception as e:
        pass


# In[5]:


print(len(train_img))
print(len(train_ships))
print(len(train_label))
print(len(test_img))
print(len(test_ships))
print(len(test_label))


# In[7]:


X_train_img = train_img
X_train_ships = train_ships
y_train = train_label
X_test_img = test_img
X_test_ships = test_ships
y_test = test_label


# In[8]:


X_train_img = np.array(X_train_img)
X_train_img = X_train_img.reshape(-1,61,61,1)
X_train_img = X_train_img.astype('float32')
X_train_ships = np.array(X_train_ships)
X_train_ships = X_train_ships.reshape(-1,10)
y_train = np.array(y_train)


# In[9]:


X_test_img = np.array(X_test_img)
X_test_img = X_test_img.reshape(-1,61,61,1)
X_test_img = X_test_img.astype('float32')
X_test_ships = np.array(X_test_ships)
X_test_ships = X_test_ships.reshape(-1,10)
y_test = np.array(y_test)


# In[32]:


#300km
ships_input = keras.Input(shape =(10,), name = "ships_layer")
img_input = keras.Input(shape =(61, 61, 1), name = "img_layer")

w = keras.layers.Conv2D(64,12) (img_input)
w = keras.layers.Conv2D(64,12)(w)
w = keras.layers.Conv2D(64,2)(w)
w = keras.layers.BatchNormalization()(w)
w = keras.activations.linear(w)
w = keras.layers.MaxPool2D(2,2)(w)
w = keras.layers.Conv2D(64, 9)(w)
w = keras.layers.Conv2D(64, 9)(w)
w = keras.layers.Conv2D(256,2)(w)
w = keras.layers.BatchNormalization()(w)
img_output1 = keras.layers.Flatten()(w)

merged_model1 = keras.layers.concatenate([img_output1, ships_input])
output_layer1 = keras.layers.Dense(256)(merged_model1)
output_layer1 = keras.layers.Dense(1, activation = 'sigmoid')(output_layer1)

new_model1 = keras.Model(inputs = [img_input, ships_input], outputs = output_layer1, name = "model_1")

new_model1.summary()

x = keras.layers.Conv2D(256, 12) (img_input)
x = keras.layers.BatchNormalization()(x)
x = keras.activations.linear(x)
x = keras.layers.MaxPool2D(2,2)(x)
x = keras.layers.Conv2D(128,2, activation = 'linear')(x)
x = keras.layers.Conv2D(128,7)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.activations.linear(x)
x = keras.layers.MaxPool2D(2,2)(x)
x = keras.layers.Conv2D(64,2, activation = 'linear')(x)
x = keras.layers.Conv2D(64,4)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.activations.linear(x)
x = keras.layers.MaxPool2D(2,2)(x)
img_output2 = keras.layers.Flatten()(x)

merged_model2 = keras.layers.concatenate([img_output2, ships_input])
output_layer2 = keras.layers.Dense(1, activation = 'sigmoid')(merged_model2)

new_model2 = keras.Model(inputs = [img_input, ships_input], outputs = output_layer2, name = "model_2")

new_model2.summary()

y = keras.layers.Conv2D(256, (10,4))(img_input)
y = keras.layers.Conv2D(256, (4,10))(y)
y = keras.layers.BatchNormalization()(y)
y = keras.activations.linear(y)
y = keras.layers.MaxPool2D(2,2)(y)
y = keras.layers.Conv2D(128,6)(y)
y = keras.layers.BatchNormalization()(y)
y = keras.activations.linear(y)
y = keras.layers.MaxPool2D(2,2)(y)
y = keras.layers.Conv2D(64,4)(y)
y = keras.layers.BatchNormalization()(y)
y = keras.activations.linear(y)
y = keras.layers.MaxPool2D(2,2)(y)
img_output3 = keras.layers.Flatten()(y)

merged_model3 = keras.layers.concatenate([img_output3, ships_input])
output_layer3 = keras.layers.Dense(1, activation = 'sigmoid')(merged_model3)

new_model3 = keras.Model(inputs = [img_input, ships_input], outputs = output_layer3, name = "model_3")

new_model3.summary()

z = keras.layers.Conv2D(256, 10)(img_input)
z = keras.layers.MaxPool2D(2,2)(z)
img_output4 = keras.layers.Flatten()(z)

merged_model4 = keras.layers.concatenate([img_output4, ships_input])
output_layer4 = keras.layers.Dense(1, activation = 'sigmoid')(merged_model4)

new_model4 = keras.Model(inputs = [img_input, ships_input], outputs = output_layer4, name = "model_4")

new_model4.summary()


# In[ ]:


new_model1.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits = False),
              metrics=['mae', 'mse'])
new_model2.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits = False),
              metrics=['mae', 'mse'])
new_model3.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits = False),
              metrics=['mae', 'mse'])
# new_model4.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits = False),
#               metrics=['mae', 'mse'])


# In[19]:


new_model1.fit([X_train_img, X_train_ships], y_train, epochs=6, batch_size=1, validation_split = 0.1)
res = new_model1.evaluate([X_test_img, X_test_ships], y_test)
print("MAE = " + str(res[0]))
print("RMSE = " + str((res[2]) ** 0.5))
preds = new_model1.predict([X_test_img, X_test_ships])
a = np.average(preds, axis = 1)
test['preds'] = a


# In[ ]:


test

