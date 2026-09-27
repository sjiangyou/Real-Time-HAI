import os
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    os.chdir(Path(__file__).parent.parent)
    # process_train_data()
    # process_test_data()
    # load_model()
    # calculate_shap_values()


def process_train_data():
    train_no_resample = pd.read_csv("IMERG/DEV/ALL_TRAIN_DATA.csv")
    train_no_resample = train_no_resample[
        ["GIS_ID", "VMAX", "VMAX_N06", "VMAX_N12", "CAT"]
    ]

    categories = train_no_resample["CAT"].unique()

    train_no_resample_TD = train_no_resample[train_no_resample["CAT"] == "TD"]
    train_no_resample_TD = train_no_resample_TD.sample(750)
    train_no_resample_TS = train_no_resample[train_no_resample["CAT"] == "TS"]
    train_no_resample_TS = train_no_resample_TS.sample(750)
    train_no_resample_Min = train_no_resample[train_no_resample["CAT"] == "Min"]
    train_no_resample_Min = train_no_resample_Min.sample(750)
    train_no_resample_Maj = train_no_resample[train_no_resample["CAT"] == "Maj"]
    train_no_resample_Maj = train_no_resample_Maj.sample(750)
    shap_train = pd.concat(
        [
            train_no_resample_TD,
            train_no_resample_TS,
            train_no_resample_Min,
            train_no_resample_Maj,
        ]
    )
    shap_train = shap_train.reset_index(drop=True)

    shap_train_img = []
    shap_train_ships = []
    shap_train_label = []

    for f in range(len(shap_train.GIS_ID)):
        filename = f"IMERG_CSV/{shap_train.GIS_ID[f]}.csv"
        try:
            temp = pd.read_csv(filename, header=None)
            if temp.shape != (121, 121):
                continue
            temp = temp[40:81]
            temp = temp.iloc[:, 40:81]
            temp = np.array(temp)
            shap_train_img.append(temp)
            lab = shap_train.VMAX[f]
            shap_train_label.append(lab)
            pvmax = np.array([shap_train.VMAX_N06[f], shap_train.VMAX_N12[f]])
            shap_train_ships.append(pvmax)
        except Exception as e:
            pass

    shap_train_img = np.array(shap_train_img)
    shap_train_img = shap_train_img.reshape(-1, 41, 41, 1)
    shap_train_img = shap_train_img.astype("float32")
    shap_train_ships = np.array(shap_train_ships)
    shap_train_ships = shap_train_ships.reshape(-1, 2)
    shap_train_label = np.array(shap_train_label)

    return shap_train_img, shap_train_ships, shap_train_label


def process_test_data():
    test = pd.read_csv("IMERG/DEV/ALL_TEST_DATA.csv")
    test = test[["GIS_ID", "VMAX", "VMAX_N06", "VMAX_N12", "CAT"]]

    test_TD = test[test["CAT"] == "TD"]
    test_TS = test[test["CAT"] == "TS"]
    test_Min = test[test["CAT"] == "Min"]
    test_Maj = test[test["CAT"] == "Maj"]
    test_TD = test_TD.sample(100)
    test_TS = test_TS.sample(100)
    test_Min = test_Min.sample(100)
    test_Maj = test_Maj.sample(100)
    shap_test = pd.concat([test_TD, test_TS, test_Min, test_Maj])

    shap_test_img = []
    shap_test_ships = []
    shap_test_label = []
    for f in range(len(shap_test.GIS_ID)):
        filename = f"IMERG_CSV/{shap_test.GIS_ID[f]}.csv"
        try:
            temp = pd.read_csv(filename, header=None)
            if temp.shape != (121, 121):
                continue
            temp = temp[40:81]
            temp = temp.iloc[:, 40:81]
            temp = np.array(temp)
            shap_test_img.append(temp)
            lab = shap_test.VMAX[f]
            shap_test_label.append(lab)
            pvmax = np.array([shap_test.VMAX_N06[f], shap_test.VMAX_N12[f]])
            shap_test_ships.append(pvmax)
        except Exception as e:
            pass
    shap_test_img = np.array(shap_test_img)
    shap_test_img = shap_test_img.reshape(-1, 41, 41, 1)
    shap_test_img = shap_test_img.astype("float64")
    shap_test_ships = np.array(shap_test_ships)
    shap_test_ships = shap_test_ships.reshape(-1, 2)
    shap_test_label = np.array(shap_test_label)

    return shap_test_img, shap_test_ships, shap_test_label


def compute_shap_values(model, background, test_inputs):
    """Calculate HIECNN image SHAP values with SHAP's PyTorch backend."""
    model = model.to(DEVICE).eval()
    background = (
        torch.from_numpy(background).to(DEVICE)
        if isinstance(background, np.ndarray)
        else background.to(DEVICE)
    )
    test_inputs = (
        torch.from_numpy(test_inputs).to(DEVICE)
        if isinstance(test_inputs, np.ndarray)
        else test_inputs.to(DEVICE)
    )
    explainer = shap.DeepExplainer(model, background)
    return explainer.shap_values(test_inputs, check_additivity=False)


if __name__ == "__main__":
    main()
