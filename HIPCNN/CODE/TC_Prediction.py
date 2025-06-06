#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Author: Sunny You
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras
import shap


# In[9]:


os.chdir('/Users/sunnyyou/Documents/Real_Time_HAI/HIPCNN/IMERG')


# In[181]:


train = pd.read_csv("DEV/P06_2018_train_resample.csv")
train = train[["GIS_ID", "DATE", "SHIPS_PER", "SHIPS_POT", "VMAX", "SHDC", "IC", "Category"]]
# train = train.drop(["VMAX_FT"], axis = 1)
train.columns = ["GIS_ID", "DATE", "PER", "POT", "VMAX", "SHDC_FT", "IC", "Category"]
train


# In[182]:


test = pd.read_csv("DEV/P06_2018_test.csv")
test = test[["GIS_ID", "DATE", "SHIPS_PER", "SHIPS_POT", "VMAX", "SHDC", "IC", "Category"]]
# test = test.drop(["VMAX_FT"], axis = 1)
test.columns = ["GIS_ID", "DATE", "PER", "POT", "VMAX", "SHDC_FT", "IC", "Category"]
test


# In[183]:


# # train = pd.concat([train, test])
# test = pd.DataFrame(np.reshape(['ATL_202313_C3_2023090718', '2023090718', 35, 54, 105, 9.9, -999, 'C3'], (1, 8)))
# test.columns = ('GIS_ID', 'DATE', 'PER', 'POT', 'VMAX', 'SHDC_FT', 'IC', 'Category')
# train = train.iloc[:, :-1]
# train = train.reset_index(drop = False)
# print(train.head(), train.shape)
# print(test.head(), test.shape)


# In[184]:


train_img = []
train_ships = []
train_label = []
test_img = []
test_ships = []
test_label = []
for f in range(len(train.GIS_ID)) :
    filename = "IMERG_CSV/" + train.GIS_ID[f] + ".csv"
    try:
        temp = pd.read_csv(filename, header = None)
        if (temp.shape != (121,121)):
            continue
        temp = temp[40:81]
        temp = temp.iloc[:, 40:81]
        temp = np.array(temp)
        train_img.append(temp)
        lab = train.IC[f] + train.VMAX[f]
        train_label.append(lab)
        ships = np.array([train.VMAX[f], train.POT[f], train.PER[f], train.SHDC_FT[f]])
        train_ships.append(ships)
    except Exception as e:
        pass

for f in range(len(test.GIS_ID)) :
    filename = "IMERG_CSV/" + test.GIS_ID[f] + ".csv"
    try:
        temp = pd.read_csv(filename, header = None)
        if (temp.shape != (121,121)):
            continue
        temp = temp[40:81]
        temp = temp.iloc[:, 40:81]
        temp = np.array(temp)
        test_img.append(temp)
        lab = test.IC[f] + test.VMAX[f]
        test_label.append(lab)
        ships = np.array([test.VMAX[f], test.POT[f], test.PER[f], test.SHDC_FT[f]])
        test_ships.append(ships)
    except Exception as e:
        pass


# In[185]:


print(len(train_img))
print(len(train_ships))
print(len(train_label))
print(len(test_img))
print(len(test_ships))
print(len(test_label))


# In[186]:


test_ships = np.float64(test_ships)
test_label = np.float64(test_label)


# In[187]:


X_train_img = train_img
X_train_ships = train_ships
y_train = train_label
X_test_img = test_img
X_test_ships = test_ships
y_test = test_label


# In[188]:


X_train_img = np.array(X_train_img)
X_train_img = X_train_img.reshape(-1,41,41,1)
X_train_img = X_train_img.astype('float32')
X_train_ships = np.array(X_train_ships)
X_train_ships = X_train_ships.reshape(-1,4)
y_train = np.array(y_train)


# In[189]:


X_test_img = np.array(X_test_img)
X_test_img = X_test_img.reshape(-1,41,41,1)
X_test_img = X_test_img.astype('float32')
X_test_ships = np.array(X_test_ships)
X_test_ships = X_test_ships.reshape(-1,4)
y_test = np.array(y_test)


# In[190]:


print(X_train_img.shape)
print(X_train_ships.shape)


# In[136]:


ships_input = keras.Input(shape =(4,), name = "ships_layer")
img_input = keras.Input(shape =(41, 41, 1), name = "img_layer")

w = keras.layers.Conv2D(64,8) (img_input)
w = keras.layers.Conv2D(64,8)(w)
w = keras.layers.Conv2D(64,1)(w)
w = keras.layers.BatchNormalization()(w)
w = keras.activations.relu(w)
w = keras.layers.MaxPool2D(2,2)(w)
w = keras.layers.Conv2D(64, 3)(w)
w = keras.layers.Conv2D(64, 3)(w)
w = keras.layers.Conv2D(256,1)(w)
w = keras.layers.BatchNormalization()(w)
img_output1 = keras.layers.Flatten()(w)

merged_model1 = keras.layers.concatenate([img_output1, ships_input])
output_layer1 = keras.layers.Dense(256)(merged_model1)
output_layer1 = keras.layers.Dense(165)(output_layer1)

new_model1 = keras.Model(inputs = [img_input, ships_input], outputs = output_layer1, name = "model_1")

new_model1.summary()

x = keras.layers.Conv2D(256, 8) (img_input)
x = keras.layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = keras.layers.MaxPool2D(2,2)(x)
x = keras.layers.Conv2D(128,1, activation = 'relu')(x)
x = keras.layers.Conv2D(128,5)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = keras.layers.MaxPool2D(2,2)(x)
x = keras.layers.Conv2D(64,1, activation = 'relu')(x)
x = keras.layers.Conv2D(64,3)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = keras.layers.MaxPool2D(2,2)(x)
img_output2 = keras.layers.Flatten()(x)

merged_model2 = keras.layers.concatenate([img_output2, ships_input])
output_layer2 = keras.layers.Dense(165)(merged_model2)

new_model2 = keras.Model(inputs = [img_input, ships_input], outputs = output_layer2, name = "model_2")

new_model2.summary()

y = keras.layers.Conv2D(256, (6,2))(img_input)
y = keras.layers.Conv2D(256, (2,6))(y)
y = keras.layers.BatchNormalization()(y)
y = keras.activations.relu(y)
y = keras.layers.MaxPool2D(2,2)(y)
y = keras.layers.Conv2D(128,4)(y)
y = keras.layers.BatchNormalization()(y)
y = keras.activations.relu(y)
y = keras.layers.MaxPool2D(2,2)(y)
y = keras.layers.Conv2D(64,3)(y)
y = keras.layers.BatchNormalization()(y)
y = keras.activations.relu(y)
y = keras.layers.MaxPool2D(2,2)(y)
img_output3 = keras.layers.Flatten()(y)

merged_model3 = keras.layers.concatenate([img_output3, ships_input])
output_layer3 = keras.layers.Dense(165)(merged_model3)

new_model3 = keras.Model(inputs = [img_input, ships_input], outputs = output_layer3, name = "model_3")

new_model3.summary()

z = keras.layers.Conv2D(256, 7)(img_input)
z = keras.layers.MaxPool2D(2,2)(z)
img_output4 = keras.layers.Flatten()(z)

merged_model4 = keras.layers.concatenate([img_output4, ships_input])
output_layer4 = keras.layers.Dense(165)(merged_model4)

new_model4 = keras.Model(inputs = [img_input, ships_input], outputs = output_layer4, name = "model_4")

new_model4.summary()


# In[34]:


# ships_input = keras.Input(shape =(4,), name = "ships_layer")
img_input = keras.Input(shape =(41, 41, 1), name = "img_layer")

w = keras.layers.Conv2D(64,8) (img_input)
w = keras.layers.Conv2D(64,8)(w)
w = keras.layers.Conv2D(64,1)(w)
w = keras.layers.BatchNormalization()(w)
w = keras.activations.relu(w)
w = keras.layers.MaxPool2D(2,2)(w)
w = keras.layers.Conv2D(64, 3)(w)
w = keras.layers.Conv2D(64, 3)(w)
w = keras.layers.Conv2D(256,1)(w)
w = keras.layers.BatchNormalization()(w)
img_output1 = keras.layers.Flatten()(w)

merged_model1 = keras.layers.concatenate([img_output1])
output_layer1 = keras.layers.Dense(256)(merged_model1)
output_layer1 = keras.layers.Dense(165)(output_layer1)

new_model1 = keras.Model(inputs = [img_input], outputs = output_layer1, name = "model_1")

new_model1.summary()

x = keras.layers.Conv2D(256, 8) (img_input)
x = keras.layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = keras.layers.MaxPool2D(2,2)(x)
x = keras.layers.Conv2D(128,1, activation = 'relu')(x)
x = keras.layers.Conv2D(128,5)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = keras.layers.MaxPool2D(2,2)(x)
x = keras.layers.Conv2D(64,1, activation = 'relu')(x)
x = keras.layers.Conv2D(64,3)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = keras.layers.MaxPool2D(2,2)(x)
img_output2 = keras.layers.Flatten()(x)

merged_model2 = keras.layers.concatenate([img_output2])
output_layer2 = keras.layers.Dense(165)(merged_model2)

new_model2 = keras.Model(inputs = [img_input], outputs = output_layer2, name = "model_2")

new_model2.summary()

y = keras.layers.Conv2D(256, (6,2))(img_input)
y = keras.layers.Conv2D(256, (2,6))(y)
y = keras.layers.BatchNormalization()(y)
y = keras.activations.relu(y)
y = keras.layers.MaxPool2D(2,2)(y)
y = keras.layers.Conv2D(128,4)(y)
y = keras.layers.BatchNormalization()(y)
y = keras.activations.relu(y)
y = keras.layers.MaxPool2D(2,2)(y)
y = keras.layers.Conv2D(64,3)(y)
y = keras.layers.BatchNormalization()(y)
y = keras.activations.relu(y)
y = keras.layers.MaxPool2D(2,2)(y)
img_output3 = keras.layers.Flatten()(y)

merged_model3 = keras.layers.concatenate([img_output3])
output_layer3 = keras.layers.Dense(165)(merged_model3)

new_model3 = keras.Model(inputs = [img_input], outputs = output_layer3, name = "model_3")

new_model3.summary()

z = keras.layers.Conv2D(256, 7)(img_input)
z = keras.layers.MaxPool2D(2,2)(z)
img_output4 = keras.layers.Flatten()(z)

merged_model4 = keras.layers.concatenate([img_output4])
output_layer4 = keras.layers.Dense(165)(merged_model4)

new_model4 = keras.Model(inputs = [img_input], outputs = output_layer4, name = "model_4")

new_model4.summary()


# In[137]:


new_model1.compile(optimizer='adam',
              loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=['mae', 'mse'])
new_model2.compile(optimizer='adam',
              loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=['mae', 'mse'])
new_model3.compile(optimizer='adam',
              loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=['mae', 'mse'])
new_model4.compile(optimizer='adam',
              loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=['mae', 'mse'])


# In[138]:


new_model3.fit([X_train_img, X_train_ships], y_train, epochs=3, batch_size=8, validation_split = 0.1)
res = new_model3.evaluate([X_test_img, X_test_ships], y_test)
print("MAE = " + str(res[0]))
print("RMSE = " + str((res[2]) ** 0.5))
preds = new_model3.predict([X_test_img, X_test_ships])
a = np.average(preds, axis = 1)
test['preds'] = a


# In[10]:


train_no_resample = pd.read_csv("DEV/P06_2018_train.csv")
train_no_resample = train_no_resample[["GIS_ID", "DATE", "SHIPS_PER", "SHIPS_POT", "VMAX", "SHDC", "IC", "Category"]]
train_no_resample.columns = ["GIS_ID", "DATE", "PER", "POT", "VMAX", "SHDC_FT", "IC", "Category"]
train_no_resample


# In[141]:


train_no_resample['Category'].value_counts()['Maj']


# In[142]:


train_no_resample_TD = train_no_resample[train_no_resample['Category'] == 'TD']
train_no_resample_TD = train_no_resample_TD.sample(90)
train_no_resample_TS = train_no_resample[train_no_resample['Category'] == 'TS']
train_no_resample_TS = train_no_resample_TS.sample(90)
train_no_resample_Min = train_no_resample[train_no_resample['Category'] == 'Min']
train_no_resample_Min = train_no_resample_Min.sample(90)
train_no_resample_Maj = train_no_resample[train_no_resample['Category'] == 'Maj']
train_no_resample_Maj = train_no_resample_Maj.sample(90)
shap_train = pd.concat([train_no_resample_TD, train_no_resample_TS, train_no_resample_Min, train_no_resample_Maj])
shap_train = shap_train.reset_index(drop = True)
shap_train


# In[143]:


shap_train_img = []
shap_train_ships = []
shap_train_label = []
for f in range(len(shap_train.GIS_ID)) :
    filename = "IMERG_CSV/" + shap_train.GIS_ID[f] + ".csv"
    try:
        temp = pd.read_csv(filename, header = None)
        if (temp.shape != (121,121)):
            continue
        temp = temp[40:81]
        temp = temp.iloc[:, 40:81]
        temp = np.array(temp)
        shap_train_img.append(temp)
        lab = shap_train.IC[f] + shap_train.VMAX[f]
        shap_train_label.append(lab)
        ships = np.array([shap_train.VMAX[f], shap_train.POT[f], shap_train.PER[f], shap_train.SHDC_FT[f]])
        shap_train_ships.append(ships)
    except Exception as e:
        pass


# In[144]:


shap_train_img = np.array(shap_train_img)
shap_train_img = shap_train_img.reshape(-1,41,41,1)
shap_train_img = shap_train_img.astype('float32')
shap_train_ships = np.array(shap_train_ships)
shap_train_ships = shap_train_ships.reshape(-1,4)
shap_train_label = np.array(shap_train_label)


# In[145]:


len(shap_train_label)


# In[146]:


#shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.linearity_1d(0)
e = shap.DeepExplainer(new_model3, [shap_train_img, shap_train_ships])
shap_values = e.shap_values([X_test_img, X_test_ships])
# shap.summary_plot(shap_values, X_test_img)
# shap.plots.beeswarm(shap_values)


# In[36]:


shap_values


# In[147]:


np.shape(shap_values[0][1])


# In[148]:


shap_values[0][1][0][3]


# In[149]:


shap_values[0][1]


# In[150]:


shap_a = []
for node in range(165):
    for images in range(360):
        shap_a.append(np.sum(shap_values[node][0][images]))

shap_b = []
for node in range(165):
    for images in range(360):
        for ships in range(4):
            shap_b.append(np.sum(shap_values[node][1][images][ships]))


# In[151]:


len(shap_b)


# In[152]:


shap_b
np.max(shap_b)

shap_VMAX = [shap_b[i] for i in range(0, len(shap_b), 4)]
shap_POT = [shap_b[i] for i in range(1, len(shap_b), 4)]
shap_PER = [shap_b[i] for i in range(2, len(shap_b), 4)]
shap_SHDC = [shap_b[i] for i in range(3, len(shap_b), 4)]


# In[178]:


for shaps in [shap_a, shap_VMAX, shap_POT, shap_PER, shap_SHDC]:
    print(np.mean(shaps))


# In[44]:


np.sum(shap_b)


# In[31]:


(np.sum(np.sum(np.abs(shap_values[0][0][0]), axis = 0), axis=0))


# In[30]:


shap_a
max(shap_a)


# In[45]:


shap.plots.force(e.expected_value[0], shap_values[0], [X_test_img[0], X_test_ships[0]])
shap.plots.beeswarm(shap_values)


# In[68]:


np.shape(shap_values[0][0][0])


# In[100]:


shap_values_nov21 = shap_values
e_nov21 = e


# In[ ]:




