# Author: Sunny You
import os
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import shap
import sklearn


def main():
    data = load_data()
    models = model_setup(*data)
    run_models(*models, *data)
    shap_analysis(models[0], data[3], data[4])


def load_data():
    train = pd.read_csv("IMERG/Model_Data/ATL_train.csv")
    train = train[
        [
            "GIS_ID",
            "DATE",
            "VMAX",
            "SHIPS_PER",
            "SHIPS_POT_Avg24h",
            "SHIPS_NOHC_Avg24h",
            "SHIPS_SHDC_Avg24h",
            "SHIPS_CFLX_Avg24h",
            "SHIPS_D200_Avg24h",
            "SHIPS_MTPW_108h",
            "SHIPS_PC2",
            "SHIPS_IR00_12h",
            "Category",
            "RI",
            "Year",
        ]
    ]
    train = train.set_axis(
        [
            "GIS_ID",
            "DATE",
            "VMAX",
            "PER",
            "POT",
            "NOHC",
            "SHDC",
            "ICDA",
            "D200",
            "TPW",
            "PC2",
            "SDBT",
            "Category",
            "RI",
            "Year",
        ],
        axis=1,
    )

    test = pd.read_csv("IMERG/Model_Data/ATL_test.csv")
    test = test[
        [
            "GIS_ID",
            "DATE",
            "VMAX",
            "SHIPS_PER",
            "SHIPS_POT_Avg24h",
            "SHIPS_NOHC_Avg24h",
            "SHIPS_SHDC_Avg24h",
            "SHIPS_CFLX_Avg24h",
            "SHIPS_D200_Avg24h",
            "SHIPS_MTPW_108h",
            "SHIPS_PC2",
            "SHIPS_IR00_12h",
            "Category",
            "RI",
            "Year",
        ]
    ]
    test = test.set_axis(
        [
            "GIS_ID",
            "DATE",
            "VMAX",
            "PER",
            "POT",
            "NOHC",
            "SHDC",
            "ICDA",
            "D200",
            "TPW",
            "PC2",
            "SDBT",
            "Category",
            "RI",
            "Year",
        ],
        axis=1,
    )

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
            temp = temp.iloc[30:91, 30:91]
            temp = np.array(temp)
            train_img.append(temp)
            lab = train.RI[f]
            train_label.append(lab)
            ships = np.array(
                [
                    train.VMAX[f],
                    train.PER[f],
                    train.POT[f],
                    train.NOHC[f],
                    train.SHDC[f],
                    train.ICDA[f],
                    train.D200[f],
                    train.TPW[f],
                    train.PC2[f],
                    train.SDBT[f],
                ]
            )
            train_ships.append(ships)
        except Exception as e:
            print(e)

    for f in range(len(test.GIS_ID)):
        filename = "IMERG_CSV/" + test.GIS_ID[f] + ".csv"
        try:
            temp = pd.read_csv(filename, header=None)
            if temp.shape != (121, 121):
                continue
            temp = temp.iloc[30:91, 30:91]
            temp = np.array(temp)
            test_img.append(temp)
            lab = test.RI[f]
            test_label.append(lab)
            ships = np.array(
                [
                    test.VMAX[f],
                    test.PER[f],
                    test.POT[f],
                    test.NOHC[f],
                    test.SHDC[f],
                    test.ICDA[f],
                    test.D200[f],
                    test.TPW[f],
                    test.PC2[f],
                    test.SDBT[f],
                ]
            )
            test_ships.append(ships)
        except Exception as e:
            pass

    print(len(train_img))
    print(len(train_ships))
    print(len(train_label))
    print(len(test_img))
    print(len(test_ships))
    print(len(test_label))

    X_train_img = train_img
    X_train_ships = train_ships
    y_train = train_label
    X_test_img = test_img
    X_test_ships = test_ships
    y_test = test_label

    X_train_img = np.array(X_train_img)
    X_train_img = X_train_img.reshape(-1, 61, 61, 1)
    X_train_img = X_train_img.astype("float32")
    X_train_ships = np.array(X_train_ships)
    X_train_ships = X_train_ships.reshape(-1, 10)
    y_train = np.array(y_train)

    X_test_img = np.array(X_test_img)
    X_test_img = X_test_img.reshape(-1, 61, 61, 1)
    X_test_img = X_test_img.astype("float32")
    X_test_ships = np.array(X_test_ships)
    X_test_ships = X_test_ships.reshape(-1, 10)
    y_test = np.array(y_test)

    return (
        X_train_img,
        X_train_ships,
        y_train,
        X_test_img,
        X_test_ships,
        y_test,
    )


def model_setup(X_train_img, X_train_ships, y_train, X_test_img, X_test_ships, y_test):
    # 300km
    ships_input = keras.Input(shape=(10,), name="ships_layer")
    img_input = keras.Input(shape=(61, 61, 1), name="img_layer")

    w = keras.layers.Conv2D(64, 12)(img_input)
    w = keras.layers.Conv2D(64, 12)(w)
    w = keras.layers.Conv2D(64, 2)(w)
    w = keras.layers.BatchNormalization()(w)
    w = keras.activations.linear(w)
    w = keras.layers.MaxPool2D(2, 2)(w)
    w = keras.layers.Conv2D(64, 9)(w)
    w = keras.layers.Conv2D(64, 9)(w)
    w = keras.layers.Conv2D(256, 2)(w)
    w = keras.layers.BatchNormalization()(w)
    img_output1 = keras.layers.Flatten()(w)

    merged_model1 = keras.layers.concatenate([img_output1, ships_input])
    output_layer1 = keras.layers.Dense(256)(merged_model1)
    output_layer1 = keras.layers.Dense(1, activation="sigmoid")(output_layer1)

    new_model1 = keras.Model(
        inputs=[img_input, ships_input], outputs=output_layer1, name="model_1"
    )

    new_model1.summary()

    x = keras.layers.Conv2D(256, 12)(img_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.activations.linear(x)
    x = keras.layers.MaxPool2D(2, 2)(x)
    x = keras.layers.Conv2D(128, 2, activation="linear")(x)
    x = keras.layers.Conv2D(128, 7)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.activations.linear(x)
    x = keras.layers.MaxPool2D(2, 2)(x)
    x = keras.layers.Conv2D(64, 2, activation="linear")(x)
    x = keras.layers.Conv2D(64, 4)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.activations.linear(x)
    x = keras.layers.MaxPool2D(2, 2)(x)
    img_output2 = keras.layers.Flatten()(x)

    merged_model2 = keras.layers.concatenate([img_output2, ships_input])
    output_layer2 = keras.layers.Dense(1, activation="sigmoid")(merged_model2)

    new_model2 = keras.Model(
        inputs=[img_input, ships_input], outputs=output_layer2, name="model_2"
    )

    new_model2.summary()

    y = keras.layers.Conv2D(256, (10, 4))(img_input)
    y = keras.layers.Conv2D(256, (4, 10))(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.activations.linear(y)
    y = keras.layers.MaxPool2D(2, 2)(y)
    y = keras.layers.Conv2D(128, 6)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.activations.linear(y)
    y = keras.layers.MaxPool2D(2, 2)(y)
    y = keras.layers.Conv2D(64, 4)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.activations.linear(y)
    y = keras.layers.MaxPool2D(2, 2)(y)
    img_output3 = keras.layers.Flatten()(y)

    merged_model3 = keras.layers.concatenate([img_output3, ships_input])
    output_layer3 = keras.layers.Dense(1, activation="sigmoid")(merged_model3)

    new_model3 = keras.Model(
        inputs=[img_input, ships_input], outputs=output_layer3, name="model_3"
    )

    new_model3.summary()

    new_model1.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["mae", "mse"],
    )
    new_model2.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["mae", "mse"],
    )
    new_model3.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["mae", "mse"],
    )

    return new_model1, new_model2, new_model3


def run_models(
    new_model1,
    new_model2,
    new_model3,
    X_train_img,
    X_train_ships,
    y_train,
    X_test_img,
    X_test_ships,
    y_test,
):
    new_model1.fit(
        [X_train_img, X_train_ships],
        y_train,
        epochs=6,
        batch_size=1,
        validation_split=0.1,
    )
    res = new_model1.evaluate([X_test_img, X_test_ships], y_test)
    preds = new_model1.predict([X_test_img, X_test_ships])
    a = np.average(preds, axis=1)
    test["preds"] = a


def shap_analysis(new_model1, X_test_img, X_test_ships):
    train_no_resample = pd.read_csv("IMERG/Model_Data/ATL_train.csv")
    train_no_resample = train_no_resample[
        [
            "GIS_ID",
            "DATE",
            "VMAX",
            "SHIPS_PER",
            "SHIPS_POT_Avg24h",
            "SHIPS_NOHC_Avg24h",
            "SHIPS_SHDC_Avg24h",
            "SHIPS_CFLX_Avg24h",
            "SHIPS_D200_Avg24h",
            "SHIPS_MTPW_108h",
            "SHIPS_PC2",
            "SHIPS_IR00_12h",
            "Category",
            "RI",
        ]
    ]
    train_no_resample = train_no_resample.set_axis(
        [
            "GIS_ID",
            "DATE",
            "VMAX",
            "PER",
            "POT",
            "NOHC",
            "SHDC",
            "ICDA",
            "D200",
            "TPW",
            "PC2",
            "SDBT",
            "Category",
            "RI",
        ],
        axis=1,
    )
    train_no_resample["RI"] = pd.to_numeric(train_no_resample["RI"], errors="coerce")

    train_no_resample["RI"].value_counts()[0]

    train_no_resample_false = train_no_resample[train_no_resample["RI"] == 0]
    train_no_resample_true = train_no_resample[train_no_resample["RI"] == 1]
    train_no_resample_false = train_no_resample_false.sample(100)
    train_no_resample_true = train_no_resample_true.sample(100)

    shap_train = pd.concat(
        [
            train_no_resample_false,
            train_no_resample_true,
        ]
    )
    shap_train = shap_train.reset_index(drop=True)

    shap_train_img = []
    shap_train_ships = []
    shap_train_label = []
    for f in range(len(shap_train.GIS_ID)):
        filename = "IMERG_Data_Old/IMERG_CSV/" + train.GIS_ID[f] + ".csv"
        try:
            temp = pd.read_csv(filename, header=None)
            if temp.shape != (121, 121):
                continue
            temp = temp[30:91, 30:91]
            temp = np.array(temp)
            train_img.append(temp)
            lab = train.RI[f]
            train_label.append(lab)
            ships = np.array(
                [
                    train.VMAX[f],
                    train.PER[f],
                    train.POT[f],
                    train.NOHC[f],
                    train.SHDC[f],
                    train.ICDA[f],
                    train.D200[f],
                    train.TPW[f],
                    train.PC2[f],
                    train.SDBT[f],
                ]
            )
            train_ships.append(ships)
        except Exception as e:
            pass

    shap_train_img = np.array(shap_train_img)
    shap_train_img = shap_train_img.reshape(-1, 61, 61, 1)
    shap_train_img = shap_train_img.astype("float32")
    shap_train_ships = np.array(shap_train_ships)
    shap_train_ships = shap_train_ships.reshape(-1, 10)
    shap_train_label = np.array(shap_train_label)

    # shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = (
        shap.explainers._deep.deep_tf.linearity_1d(0)
    )
    e = shap.DeepExplainer(new_model1, [shap_train_img, shap_train_ships])
    shap_values = e.shap_values([X_test_img, X_test_ships])
    # shap.summary_plot(shap_values, X_test_img)
    # shap.plots.beeswarm(shap_values)

    shap_image = []
    for node in range(1):
        for images in range(600):
            shap_image.append(np.sum(shap_values[node][0][images]))

    shap_ships = []
    for node in range(1):
        for images in range(360):
            for ships in range(10):
                shap_ships.append(np.sum(shap_values[node][1][images][ships]))

    shap_VMAX = [shap_ships[i] for i in range(0, len(shap_ships), 4)]
    shap_POT = [shap_ships[i] for i in range(1, len(shap_ships), 4)]
    shap_PER = [shap_ships[i] for i in range(2, len(shap_ships), 4)]
    shap_NOHC = [shap_ships[i] for i in range(3, len(shap_ships), 4)]
    shap_SHDC = [shap_ships[i] for i in range(4, len(shap_ships), 4)]
    shap_ICDA = [shap_ships[i] for i in range(5, len(shap_ships), 4)]
    shap_D200 = [shap_ships[i] for i in range(6, len(shap_ships), 4)]
    shap_TPW = [shap_ships[i] for i in range(7, len(shap_ships), 4)]
    shap_PC2 = [shap_ships[i] for i in range(8, len(shap_ships), 4)]
    shap_SDBT = [shap_ships[i] for i in range(9, len(shap_ships), 4)]

    for shaps in [
        shap_image,
        shap_VMAX,
        shap_POT,
        shap_PER,
        shap_NOHC,
        shap_SHDC,
        shap_ICDA,
        shap_D200,
        shap_TPW,
        shap_PC2,
        shap_SDBT,
    ]:
        print(f"Mean: {np.mean(shaps)}")
        print(f"Median: {np.median(shaps)}")
        print(f"Max: {np.max(shaps)}")
        print(f"Min: {np.min(shaps)}")
        print("\n")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    main()
