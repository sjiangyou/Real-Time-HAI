# Author: Sunny You
import os
import pandas as pd
import numpy as np
import shap
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def define_models():
    model_1 = (
        nn.Sequential(
            nn.Conv2d(1, 64, 8),
            nn.Conv2d(64, 64, 8),
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.Conv2d(64, 64, 3),
            nn.Conv2d(64, 256, 1),
            nn.BatchNorm2d(256),
        ),
        nn.Sequential(nn.Linear(256 * 9 * 9 + 4, 256), nn.Linear(256, 165)),
    )
    model_2 = (
        nn.Sequential(
            nn.Conv2d(1, 256, 8),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ),
        nn.Linear(64 * 2 * 2 + 4, 165),
    )
    model_3 = (
        nn.Sequential(
            nn.Conv2d(1, 256, (6, 2)),
            nn.Conv2d(256, 256, (2, 6)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 128, 4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ),
        nn.Linear(64 * 2 * 2 + 4, 165),
    )
    model_4 = (
        nn.Sequential(nn.Conv2d(1, 256, 7), nn.MaxPool2d(2, 2)),
        nn.Linear(256 * 17 * 17 + 4, 165),
    )
    return [
        (image.to(DEVICE), output.to(DEVICE))
        for image, output in [model_1, model_2, model_3, model_4]
    ]


os.chdir("/Users/sunnyyou/Documents/Real_Time_HAI/HIPCNN/IMERG")

train = pd.read_csv("DEV/P06_2018_train_resample.csv")
train = train[
    ["GIS_ID", "DATE", "SHIPS_PER", "SHIPS_POT", "VMAX", "SHDC", "IC", "Category"]
]
# train = train.drop(["VMAX_FT"], axis = 1)
train.columns = ["GIS_ID", "DATE", "PER", "POT", "VMAX", "SHDC_FT", "IC", "Category"]

test = pd.read_csv("DEV/P06_2018_test.csv")
test = test[
    ["GIS_ID", "DATE", "SHIPS_PER", "SHIPS_POT", "VMAX", "SHDC", "IC", "Category"]
]
# test = test.drop(["VMAX_FT"], axis = 1)
test.columns = ["GIS_ID", "DATE", "PER", "POT", "VMAX", "SHDC_FT", "IC", "Category"]


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
for f in range(len(train.GIS_ID)):
    filename = "IMERG_CSV/" + train.GIS_ID[f] + ".csv"
    try:
        temp = pd.read_csv(filename, header=None)
        if temp.shape != (121, 121):
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

for f in range(len(test.GIS_ID)):
    filename = "IMERG_CSV/" + test.GIS_ID[f] + ".csv"
    try:
        temp = pd.read_csv(filename, header=None)
        if temp.shape != (121, 121):
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
X_train_img = X_train_img.reshape(-1, 41, 41, 1)
X_train_img = X_train_img.astype("float32")
X_train_ships = np.array(X_train_ships)
X_train_ships = X_train_ships.reshape(-1, 4)
y_train = np.array(y_train)


# In[189]:


X_test_img = np.array(X_test_img)
X_test_img = X_test_img.reshape(-1, 41, 41, 1)
X_test_img = X_test_img.astype("float32")
X_test_ships = np.array(X_test_ships)
X_test_ships = X_test_ships.reshape(-1, 4)
y_test = np.array(y_test)


# In[190]:


print(X_train_img.shape)
print(X_train_ships.shape)


# In[136]:


new_model1, new_model2, new_model3, new_model4 = define_models()


def model_forward(model, image, ships):
    if image.shape[1] != 1:
        image = image.permute(0, 3, 1, 2)
    features = torch.flatten(model[0](image.float()), start_dim=1)
    return model[1](torch.cat((features, ships.float()), dim=1))


# In[138]:


optimizer = torch.optim.Adam(
    list(new_model3[0].parameters()) + list(new_model3[1].parameters())
)
loss_function = nn.L1Loss()
train_img = torch.from_numpy(X_train_img).to(DEVICE)
train_ships = torch.from_numpy(X_train_ships).to(DEVICE)
train_labels = torch.from_numpy(y_train.astype("float32")).to(DEVICE)
for epoch in range(3):
    new_model3[0].train()
    new_model3[1].train()
    optimizer.zero_grad()
    predictions = model_forward(new_model3, train_img, train_ships)
    loss = loss_function(predictions, train_labels)
    loss.backward()
    optimizer.step()

new_model3[0].eval()
new_model3[1].eval()
with torch.no_grad():
    predictions = model_forward(
        new_model3,
        torch.from_numpy(X_test_img).to(DEVICE),
        torch.from_numpy(X_test_ships).to(DEVICE),
    )
    res = loss_function(
        predictions, torch.from_numpy(y_test.astype("float32")).to(DEVICE)
    )
print("MAE = " + str(res.item()))
print(
    "RMSE = "
    + str(
        torch.mean(
            (predictions - torch.from_numpy(y_test.astype("float32")).to(DEVICE)) ** 2
        )
        .sqrt()
        .item()
    )
)
preds = predictions.cpu().numpy()
a = np.average(preds, axis=1)
test["preds"] = a


train_no_resample = pd.read_csv("DEV/P06_2018_train.csv")
train_no_resample = train_no_resample[
    ["GIS_ID", "DATE", "SHIPS_PER", "SHIPS_POT", "VMAX", "SHDC", "IC", "Category"]
]
train_no_resample.columns = [
    "GIS_ID",
    "DATE",
    "PER",
    "POT",
    "VMAX",
    "SHDC_FT",
    "IC",
    "Category",
]
train_no_resample


# In[141]:


train_no_resample["Category"].value_counts()["Maj"]


# In[142]:


train_no_resample_TD = train_no_resample[train_no_resample["Category"] == "TD"]
train_no_resample_TD = train_no_resample_TD.sample(90)
train_no_resample_TS = train_no_resample[train_no_resample["Category"] == "TS"]
train_no_resample_TS = train_no_resample_TS.sample(90)
train_no_resample_Min = train_no_resample[train_no_resample["Category"] == "Min"]
train_no_resample_Min = train_no_resample_Min.sample(90)
train_no_resample_Maj = train_no_resample[train_no_resample["Category"] == "Maj"]
train_no_resample_Maj = train_no_resample_Maj.sample(90)
shap_train = pd.concat(
    [
        train_no_resample_TD,
        train_no_resample_TS,
        train_no_resample_Min,
        train_no_resample_Maj,
    ]
)
shap_train = shap_train.reset_index(drop=True)
shap_train


# In[143]:


shap_train_img = []
shap_train_ships = []
shap_train_label = []
for f in range(len(shap_train.GIS_ID)):
    filename = "IMERG_CSV/" + shap_train.GIS_ID[f] + ".csv"
    try:
        temp = pd.read_csv(filename, header=None)
        if temp.shape != (121, 121):
            continue
        temp = temp[40:81]
        temp = temp.iloc[:, 40:81]
        temp = np.array(temp)
        shap_train_img.append(temp)
        lab = shap_train.IC[f] + shap_train.VMAX[f]
        shap_train_label.append(lab)
        ships = np.array(
            [
                shap_train.VMAX[f],
                shap_train.POT[f],
                shap_train.PER[f],
                shap_train.SHDC_FT[f],
            ]
        )
        shap_train_ships.append(ships)
    except Exception as e:
        pass


# In[144]:


shap_train_img = np.array(shap_train_img)
shap_train_img = shap_train_img.reshape(-1, 41, 41, 1)
shap_train_img = shap_train_img.astype("float32")
shap_train_ships = np.array(shap_train_ships)
shap_train_ships = shap_train_ships.reshape(-1, 4)
shap_train_label = np.array(shap_train_label)


# In[145]:


len(shap_train_label)


# In[146]:


e = shap.DeepExplainer(
    model_forward,
    [
        torch.from_numpy(shap_train_img).to(DEVICE),
        torch.from_numpy(shap_train_ships).to(DEVICE),
    ],
)
shap_values = e.shap_values(
    [
        torch.from_numpy(X_test_img).to(DEVICE),
        torch.from_numpy(X_test_ships).to(DEVICE),
    ],
    check_additivity=False,
)
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


(np.sum(np.sum(np.abs(shap_values[0][0][0]), axis=0), axis=0))


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
