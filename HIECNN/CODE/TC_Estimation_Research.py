import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


def main():
    os.chdir(Path(__file__).parent.parent)
    train_data = process_train_data()
    test_data = process_test_data()
    models = define_models()
    train_models(models, train_data, test_data)


def process_train_data():
    print("Processing training data...")

    train = pd.read_csv("IMERG/DEV/ALL_TRAIN_DATA_RESAMPLE.csv")
    train = train[["GIS_ID", "VMAX", "VMAX_N06", "VMAX_N12"]]

    train_img = []
    train_vmax = []
    train_label = []
    for f in range(len(train.GIS_ID)):
        filename = f"IMERG_CSV/{train.GIS_ID[f]}.csv"
        try:
            image = pd.read_csv(filename, header=None)
            if image.shape != (121, 121):
                continue
            image = image.iloc[40:81, 40:81]
            image = np.array(image)
            train_img.append(image)
            lab = train.VMAX[f]
            train_label.append(lab)
            pvmax = np.array([train.VMAX_N06[f], train.VMAX_N12[f]])
            train_vmax.append(pvmax)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    print("Training data processed.")
    print(
        f"There are {len(train_img)} images, {len(train_vmax)} intensity values, and {len(train_label)} labels."
    )

    train_img = np.array(train_img)
    train_img = train_img.reshape(-1, 41, 41, 1)
    train_img = train_img.astype("float64")
    train_vmax = np.array(train_vmax)
    train_vmax = train_vmax.reshape(-1, 2)
    train_label = np.array(train_label)

    return train_img, train_vmax, train_label


def process_test_data():
    print("Processing test data...")

    test = pd.read_csv("IMERG/DEV/ALL_TEST_DATA.csv")
    test = test[["GIS_ID", "VMAX", "VMAX_N06", "VMAX_N12"]]

    test_img = []
    test_vmax = []
    test_label = []
    for f in range(len(test.GIS_ID)):
        filename = f"IMERG_CSV/{test.GIS_ID[f]}.csv"
        try:
            image = pd.read_csv(filename, header=None)
            if image.shape != (121, 121):
                continue
            image = image.iloc[40:81, 40:81]
            image = np.array(image)
            test_img.append(image)
            lab = test.VMAX[f]
            test_label.append(lab)
            pvmax = np.array([test.VMAX_N06[f], test.VMAX_N12[f]])
            test_vmax.append(pvmax)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    print("Test data processed.")
    print(
        f"There are {len(test_img)} images, {len(test_vmax)} intensity values, and {len(test_label)} labels."
    )

    test_img = np.array(test_img)
    test_img = test_img.reshape(-1, 41, 41, 1)
    test_img = test_img.astype("float64")
    test_vmax = np.array(test_vmax)
    test_vmax = test_vmax.reshape(-1, 2)
    test_label = np.array(test_label)

    return test_img, test_vmax, test_label


def define_models():

    # 300km
    # vmax_input = keras.Input(shape=(2,), name="vmax_layer")
    # img_input = keras.Input(shape=(61, 61, 1), name="img_layer")
    #
    # model_1 = keras.layers.Conv2D(64, 12)(img_input)
    # model_1 = keras.layers.Conv2D(64, 12)(model_1)
    # model_1 = keras.layers.Conv2D(64, 2)(model_1)
    # model_1 = keras.layers.BatchNormalization()(model_1)
    # model_1 = keras.activations.linear(model_1)
    # model_1 = keras.layers.MaxPool2D(2, 2)(model_1)
    # model_1 = keras.layers.Conv2D(64, 9)(model_1)
    # model_1 = keras.layers.Conv2D(64, 9)(model_1)
    # model_1 = keras.layers.Conv2D(256, 2)(model_1)
    # model_1 = keras.layers.BatchNormalization()(model_1)
    # img_output1 = keras.layers.Flatten()(model_1)
    #
    # merged_model1 = keras.layers.concatenate([img_output1, vmax_input])
    # output_layer1 = keras.layers.Dense(256)(merged_model1)
    # output_layer1 = keras.layers.Dense(170, activation="linear")(output_layer1)
    #
    # new_model1 = keras.Model(
    #     inputs=[img_input, vmax_input], outputs=output_layer1, name="model_1"
    # )
    #
    # new_model1.summary()
    #
    # model_2 = keras.layers.Conv2D(256, 12)(img_input)
    # model_2 = keras.layers.BatchNormalization()(model_2)
    # model_2 = keras.activations.linear(model_2)
    # model_2 = keras.layers.MaxPool2D(2, 2)(model_2)
    # model_2 = keras.layers.Conv2D(128, 2, activation="linear")(model_2)
    # model_2 = keras.layers.Conv2D(128, 7)(model_2)
    # model_2 = keras.layers.BatchNormalization()(model_2)
    # model_2 = keras.activations.linear(model_2)
    # model_2 = keras.layers.MaxPool2D(2, 2)(model_2)
    # model_2 = keras.layers.Conv2D(64, 2, activation="linear")(model_2)
    # model_2 = keras.layers.Conv2D(64, 4)(model_2)
    # model_2 = keras.layers.BatchNormalization()(model_2)
    # model_2 = keras.activations.linear(model_2)
    # model_2 = keras.layers.MaxPool2D(2, 2)(model_2)
    # img_output2 = keras.layers.Flatten()(model_2)
    #
    # merged_model2 = keras.layers.concatenate([img_output2, vmax_input])
    # output_layer2 = keras.layers.Dense(170, activation="linear")(merged_model2)
    #
    # new_model2 = keras.Model(
    #     inputs=[img_input, vmax_input], outputs=output_layer2, name="model_2"
    # )
    #
    # new_model2.summary()
    #
    # model_3 = keras.layers.Conv2D(256, (10, 4))(img_input)
    # model_3 = keras.layers.Conv2D(256, (4, 10))(model_3)
    # model_3 = keras.layers.BatchNormalization()(model_3)
    # model_3 = keras.activations.linear(model_3)
    # model_3 = keras.layers.MaxPool2D(2, 2)(model_3)
    # model_3 = keras.layers.Conv2D(128, 6)(model_3)
    # model_3 = keras.layers.BatchNormalization()(model_3)
    # model_3 = keras.activations.linear(model_3)
    # model_3 = keras.layers.MaxPool2D(2, 2)(model_3)
    # model_3 = keras.layers.Conv2D(64, 4)(model_3)
    # model_3 = keras.layers.BatchNormalization()(model_3)
    # model_3 = keras.activations.linear(model_3)
    # model_3 = keras.layers.MaxPool2D(2, 2)(model_3)
    # img_output3 = keras.layers.Flatten()(model_3)
    #
    # merged_model3 = keras.layers.concatenate([img_output3, vmax_input])
    # output_layer3 = keras.layers.Dense(170, activation="linear")(merged_model3)
    #
    # new_model3 = keras.Model(
    #     inputs=[img_input, vmax_input], outputs=output_layer3, name="model_3"
    # )
    #
    # new_model3.summary()

    # 200km
    vmax_input = keras.Input(shape=(2,), name="vmax_layer")
    img_input = keras.Input(shape=(41, 41, 1), name="img_layer")

    model_1 = keras.layers.Conv2D(64, 8)(img_input)
    model_1 = keras.layers.Conv2D(64, 8)(model_1)
    model_1 = keras.layers.Conv2D(64, 1)(model_1)
    model_1 = keras.layers.BatchNormalization()(model_1)
    model_1 = keras.activations.relu(model_1)
    model_1 = keras.layers.MaxPool2D(2, 2)(model_1)
    model_1 = keras.layers.Conv2D(64, 3)(model_1)
    model_1 = keras.layers.Conv2D(64, 3)(model_1)
    model_1 = keras.layers.Conv2D(256, 1)(model_1)
    model_1 = keras.layers.BatchNormalization()(model_1)
    img_output1 = keras.layers.Flatten()(model_1)

    merged_model1 = keras.layers.concatenate([img_output1, vmax_input])
    output_layer1 = keras.layers.Dense(256)(merged_model1)
    output_layer1 = keras.layers.Dense(170)(output_layer1)

    new_model1 = keras.Model(
        inputs=[img_input, vmax_input], outputs=output_layer1, name="model_1"
    )

    # new_model1.summary()

    model_2 = keras.layers.Conv2D(256, 8)(img_input)
    model_2 = keras.layers.BatchNormalization()(model_2)
    model_2 = keras.activations.relu(model_2)
    model_2 = keras.layers.MaxPool2D(2, 2)(model_2)
    model_2 = keras.layers.Conv2D(128, 1, activation="relu")(model_2)
    model_2 = keras.layers.Conv2D(128, 5)(model_2)
    model_2 = keras.layers.BatchNormalization()(model_2)
    model_2 = keras.activations.relu(model_2)
    model_2 = keras.layers.MaxPool2D(2, 2)(model_2)
    model_2 = keras.layers.Conv2D(64, 1, activation="relu")(model_2)
    model_2 = keras.layers.Conv2D(64, 3)(model_2)
    model_2 = keras.layers.BatchNormalization()(model_2)
    model_2 = keras.activations.relu(model_2)
    model_2 = keras.layers.MaxPool2D(2, 2)(model_2)
    img_output2 = keras.layers.Flatten()(model_2)

    merged_model2 = keras.layers.concatenate([img_output2, vmax_input])
    output_layer2 = keras.layers.Dense(170)(merged_model2)

    new_model2 = keras.Model(
        inputs=[img_input, vmax_input], outputs=output_layer2, name="model_2"
    )

    # new_model2.summary()

    model_3 = keras.layers.Conv2D(256, (6, 2))(img_input)
    model_3 = keras.layers.Conv2D(256, (2, 6))(model_3)
    model_3 = keras.layers.BatchNormalization()(model_3)
    model_3 = keras.activations.relu(model_3)
    model_3 = keras.layers.MaxPool2D(2, 2)(model_3)
    model_3 = keras.layers.Conv2D(128, 4)(model_3)
    model_3 = keras.layers.BatchNormalization()(model_3)
    model_3 = keras.activations.relu(model_3)
    model_3 = keras.layers.MaxPool2D(2, 2)(model_3)
    model_3 = keras.layers.Conv2D(64, 3)(model_3)
    model_3 = keras.layers.BatchNormalization()(model_3)
    model_3 = keras.activations.relu(model_3)
    model_3 = keras.layers.MaxPool2D(2, 2)(model_3)
    img_output3 = keras.layers.Flatten()(model_3)

    merged_model3 = keras.layers.concatenate([img_output3, vmax_input])
    output_layer3 = keras.layers.Dense(170)(merged_model3)

    new_model3 = keras.Model(
        inputs=[img_input, vmax_input], outputs=output_layer3, name="model_3"
    )

    # new_model3.summary()

    print("Models defined.")
    return [new_model1, new_model2, new_model3]


def train_models(models, train_data, test_data):
    class OutputPredictions(keras.callbacks.Callback):
        def __init__(self, batch_size, id):
            super().__init__()
            self.batch_size = batch_size
            self.id = id

        # def on_batch_end(self, batch, logs=None):
        # pass

        def on_epoch_end(self, epoch, logs=None):
            results = self.model.evaluate([test_data[0], test_data[1]], test_data[2])
            mae, rmse = results[0], results[2] ** 0.5
            predictions = model.predict([test_data[0], test_data[1]])
            predictions = np.average(predictions, axis=1)
            self.write_results(
                "IMERG/DEV/ALL_TEST_DATA.csv",
                predictions,
                epoch,
            )
            with open(f"OVERALL_RESULTS.csv", "a") as f:
                f.write(f"{id},{epoch},{self.batch_size},{mae},{rmse}\n")
            tf.keras.models.save_model(
                self.model, f"MODELS/MODEL{id}_EPOCHS{epoch}_BATCH{self.batch_size}"
            )

        def write_results(self, test_csv_path, predictions, epoch):
            os.chdir(Path(__file__).parent.parent)
            csv = pd.read_csv(test_csv_path)
            csv["preds"] = predictions
            csv.to_csv(f"OUTPUT/MODEL{self.id}_RESULTS{epoch}_{self.batch_size}.csv")

    os.makedirs("MODELS", exist_ok=True)
    os.makedirs("OUTPUT", exist_ok=True)

    batches = [1, 2, 4, 8, 16]
    for bsize in batches:
        for id, model in enumerate(models):
            model.compile(
                optimizer="adam",
                loss=tf.keras.losses.MeanAbsoluteError(),
                metrics=["mae", "mse"],
            )
            print(f"Training model {id + 1} with batch size {bsize}.")
            model.fit(
                [train_data[0], train_data[1]],
                train_data[2],
                epochs=50,
                batch_size=bsize,
                callbacks=[
                    OutputPredictions(bsize, id + 1),
                    keras.callbacks.EarlyStopping(monitor="loss", patience=10),
                ],
            )
            tf.keras.backend.clear_session(free_memory=True)


if __name__ == "__main__":
    main()
