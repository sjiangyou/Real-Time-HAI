# Author: Sunny You
import os
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
EPOCHS = 6
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    gpu_test()

    data = load_data()
    models = model_setup(*data)

    run_models(*models, *data)

    # Preserve the original behavior: SHAP analysis is run for model 1.
    shap_analysis(models[0], data[3], data[4])


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


def gpu_test():
    gpu_available = torch.cuda.is_available()
    print(f"Is GPU available?: {gpu_available}")

    if gpu_available:
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU ID: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


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
        filename = "IMERG_CSV/" + train.GIS_ID.iloc[f] + ".csv"

        try:
            temp = pd.read_csv(filename, header=None)

            if temp.shape != (121, 121):
                continue

            temp = temp.iloc[30:91, 30:91]
            temp = np.array(temp)

            train_img.append(temp)
            train_label.append(train.RI.iloc[f])

            ships = np.array(
                [
                    train.VMAX.iloc[f],
                    train.PER.iloc[f],
                    train.POT.iloc[f],
                    train.NOHC.iloc[f],
                    train.SHDC.iloc[f],
                    train.ICDA.iloc[f],
                    train.D200.iloc[f],
                    train.TPW.iloc[f],
                    train.PC2.iloc[f],
                    train.SDBT.iloc[f],
                ]
            )
            train_ships.append(ships)

        except Exception as e:
            print(e)

    for f in range(len(test.GIS_ID)):
        filename = "IMERG_CSV/" + test.GIS_ID.iloc[f] + ".csv"

        try:
            temp = pd.read_csv(filename, header=None)

            if temp.shape != (121, 121):
                continue

            temp = temp.iloc[30:91, 30:91]
            temp = np.array(temp)

            test_img.append(temp)
            test_label.append(test.RI.iloc[f])

            ships = np.array(
                [
                    test.VMAX.iloc[f],
                    test.PER.iloc[f],
                    test.POT.iloc[f],
                    test.NOHC.iloc[f],
                    test.SHDC.iloc[f],
                    test.ICDA.iloc[f],
                    test.D200.iloc[f],
                    test.TPW.iloc[f],
                    test.PC2.iloc[f],
                    test.SDBT.iloc[f],
                ]
            )
            test_ships.append(ships)

        except Exception:
            pass

    print(len(train_img))
    print(len(train_ships))
    print(len(train_label))
    print(len(test_img))
    print(len(test_ships))
    print(len(test_label))

    X_train_img = np.array(train_img)
    X_train_img = X_train_img.reshape(-1, 61, 61, 1).astype("float32")

    # Keras uses NHWC; PyTorch Conv2d uses NCHW.
    X_train_img = np.transpose(X_train_img, (0, 3, 1, 2))

    X_train_ships = np.array(train_ships).reshape(-1, 10).astype("float32")
    y_train = np.array(train_label).astype("float32")

    X_test_img = np.array(test_img)
    X_test_img = X_test_img.reshape(-1, 61, 61, 1).astype("float32")
    X_test_img = np.transpose(X_test_img, (0, 3, 1, 2))

    X_test_ships = np.array(test_ships).reshape(-1, 10).astype("float32")
    y_test = np.array(test_label).astype("float32")

    return (
        X_train_img,
        X_train_ships,
        y_train,
        X_test_img,
        X_test_ships,
        y_test,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Model1(nn.Module):
    """
    Keras model_1 converted to PyTorch.

    Input:
        image: (N, 1, 61, 61)
        ships: (N, 10)

    Output:
        (N, 1), sigmoid probability
    """

    def __init__(self):
        super().__init__()

        self.image_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=12),
            nn.Conv2d(64, 64, kernel_size=12),
            nn.Conv2d(64, 64, kernel_size=2),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=9),
            nn.Conv2d(64, 64, kernel_size=9),
            nn.Conv2d(64, 256, kernel_size=2),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
        )

        # Image output: 256 x 2 x 2 = 1024
        # + 10 SHIPS variables = 1034
        self.fc1 = nn.Linear(1024 + 10, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, image, ships):
        x = self.image_layers(image)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, ships), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)

        return torch.sigmoid(x)


class Model2(nn.Module):
    """
    Keras model_2 converted to PyTorch.
    """

    def __init__(self):
        super().__init__()

        self.image_layers = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=12),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=2),
            nn.Conv2d(128, 128, kernel_size=7),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=4),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Image output: 64 x 2 x 2 = 256
        # + 10 SHIPS variables = 266
        self.fc = nn.Linear(256 + 10, 1)

    def forward(self, image, ships):
        x = self.image_layers(image)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, ships), dim=1)
        x = self.fc(x)

        return torch.sigmoid(x)


class Model3(nn.Module):
    """
    Keras model_3 converted to PyTorch.
    """

    def __init__(self):
        super().__init__()

        self.image_layers = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(10, 4)),
            nn.Conv2d(256, 256, kernel_size=(4, 10)),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=6),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=4),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Image output: 64 x 3 x 3 = 576
        # + 10 SHIPS variables = 586
        self.fc = nn.Linear(576 + 10, 1)

    def forward(self, image, ships):
        x = self.image_layers(image)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, ships), dim=1)
        x = self.fc(x)

        return torch.sigmoid(x)


def model_setup(
    X_train_img,
    X_train_ships,
    y_train,
    X_test_img,
    X_test_ships,
    y_test,
):
    new_model1 = Model1().to(DEVICE)
    new_model2 = Model2().to(DEVICE)
    new_model3 = Model3().to(DEVICE)

    print(new_model1)
    print(new_model2)
    print(new_model3)

    return new_model1, new_model2, new_model3


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def binary_accuracy(preds, targets):
    predictions = (preds >= 0.5).float()
    return (predictions == targets).float().mean().item()


def evaluate_model(model, data_loader, criterion):
    model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_mse = 0.0
    total_count = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, ships, labels in data_loader:
            images = images.to(DEVICE)
            ships = ships.to(DEVICE)
            labels = labels.to(DEVICE).view(-1, 1)

            preds = model(images, ships)
            loss = criterion(preds, labels)

            batch_size = labels.shape[0]

            total_loss += loss.item() * batch_size
            total_mae += torch.abs(preds - labels).sum().item()
            total_mse += ((preds - labels) ** 2).sum().item()
            total_count += batch_size

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return (
        total_loss / total_count,
        total_mae / total_count,
        total_mse / total_count,
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_labels, axis=0),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(model, X_train_img, X_train_ships, y_train):
    dataset = TensorDataset(
        torch.from_numpy(X_train_img),
        torch.from_numpy(X_train_ships),
        torch.from_numpy(y_train),
    )

    validation_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - validation_size

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        running_count = 0

        for images, ships, labels in train_loader:
            images = images.to(DEVICE)
            ships = ships.to(DEVICE)
            labels = labels.to(DEVICE).view(-1, 1)

            optimizer.zero_grad()

            predictions = model(images, ships)
            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.shape[0]
            running_loss += loss.item() * batch_size
            running_count += batch_size

        train_loss = running_loss / running_count

        val_loss, val_mae, val_mse, val_preds, val_labels = evaluate_model(
            model,
            validation_loader,
            criterion,
        )

        val_accuracy = binary_accuracy(
            torch.from_numpy(val_preds),
            torch.from_numpy(val_labels),
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MAE: {val_mae:.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Val Accuracy: {val_accuracy:.6f}"
        )

    return model


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


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
    # The original Keras code only trains/evaluates model 1 here.
    # That behavior is preserved.
    new_model1 = train_model(
        new_model1,
        X_train_img,
        X_train_ships,
        y_train,
    )

    test_dataset = TensorDataset(
        torch.from_numpy(X_test_img),
        torch.from_numpy(X_test_ships),
        torch.from_numpy(y_test),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    criterion = nn.BCELoss()

    test_loss, test_mae, test_mse, preds, labels = evaluate_model(
        new_model1,
        test_loader,
        criterion,
    )

    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test MSE: {test_mse:.6f}")

    preds = np.average(preds, axis=1)

    print("Test predictions:")
    print(preds)

    return new_model1, preds


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------


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

    train_no_resample["RI"] = pd.to_numeric(
        train_no_resample["RI"],
        errors="coerce",
    )

    print(train_no_resample["RI"].value_counts()[0])

    train_no_resample_false = train_no_resample[train_no_resample["RI"] == 0]
    train_no_resample_true = train_no_resample[train_no_resample["RI"] == 1]

    train_no_resample_false = train_no_resample_false.sample(
        100,
        random_state=42,
    )
    train_no_resample_true = train_no_resample_true.sample(
        100,
        random_state=42,
    )

    shap_train = pd.concat(
        [
            train_no_resample_false,
            train_no_resample_true,
        ]
    ).reset_index(drop=True)

    shap_train_img = []
    shap_train_ships = []
    shap_train_label = []

    for f in range(len(shap_train.GIS_ID)):
        filename = "IMERG_CSV/" + shap_train.GIS_ID.iloc[f] + ".csv"

        try:
            temp = pd.read_csv(filename, header=None)

            if temp.shape != (121, 121):
                continue

            temp = temp.iloc[30:91, 30:91]
            temp = np.array(temp)

            shap_train_img.append(temp)
            shap_train_label.append(shap_train.RI.iloc[f])

            ships = np.array(
                [
                    shap_train.VMAX.iloc[f],
                    shap_train.PER.iloc[f],
                    shap_train.POT.iloc[f],
                    shap_train.NOHC.iloc[f],
                    shap_train.SHDC.iloc[f],
                    shap_train.ICDA.iloc[f],
                    shap_train.D200.iloc[f],
                    shap_train.TPW.iloc[f],
                    shap_train.PC2.iloc[f],
                    shap_train.SDBT.iloc[f],
                ]
            )

            shap_train_ships.append(ships)

        except Exception:
            pass

    shap_train_img = np.array(shap_train_img).astype("float32")
    shap_train_img = shap_train_img.reshape(-1, 61, 61, 1)
    shap_train_img = np.transpose(shap_train_img, (0, 3, 1, 2))

    shap_train_ships = np.array(shap_train_ships).astype("float32")
    shap_train_ships = shap_train_ships.reshape(-1, 10)

    shap_train_label = np.array(shap_train_label)

    if len(shap_train_img) == 0:
        print("No SHAP background images were loaded.")
        return

    # SHAP's PyTorch DeepExplainer accepts a list of tensor inputs.
    new_model1.eval()

    background_img = torch.from_numpy(shap_train_img).to(DEVICE)
    background_ships = torch.from_numpy(shap_train_ships).to(DEVICE)

    test_img = torch.from_numpy(X_test_img).to(DEVICE)
    test_ships = torch.from_numpy(X_test_ships).to(DEVICE)

    explainer = shap.DeepExplainer(new_model1, [background_img, background_ships])

    shap_values = explainer.shap_values([test_img, test_ships], check_additivity=False)

    # SHAP has returned different container shapes across versions.
    # Normalize the result enough to handle the common PyTorch formats.
    if isinstance(shap_values, list):
        if len(shap_values) == 1 and isinstance(shap_values[0], list):
            shap_values = shap_values[0]

        if len(shap_values) == 2:
            shap_img_values = np.asarray(shap_values[0])
            shap_ship_values = np.asarray(shap_values[1])
        else:
            print("Unexpected SHAP output format.")
            return

    else:
        print("Unexpected SHAP output format.")
        return

    # Remove the final singleton output dimension when present.
    if shap_img_values.ndim == 5 and shap_img_values.shape[-1] == 1:
        shap_img_values = shap_img_values[..., 0]

    if shap_ship_values.ndim == 3 and shap_ship_values.shape[-1] == 1:
        shap_ship_values = shap_ship_values[..., 0]

    shap_image = [np.sum(sample) for sample in shap_img_values]

    shap_ships = [np.sum(sample) for sample in shap_ship_values]

    # The original code attempted to extract these ten variables.
    # Here they are indexed by feature instead of using a stride of four.
    shap_VMAX = shap_ship_values[:, 0].flatten()
    shap_PER = shap_ship_values[:, 1].flatten()
    shap_POT = shap_ship_values[:, 2].flatten()
    shap_NOHC = shap_ship_values[:, 3].flatten()
    shap_SHDC = shap_ship_values[:, 4].flatten()
    shap_ICDA = shap_ship_values[:, 5].flatten()
    shap_D200 = shap_ship_values[:, 6].flatten()
    shap_TPW = shap_ship_values[:, 7].flatten()
    shap_PC2 = shap_ship_values[:, 8].flatten()
    shap_SDBT = shap_ship_values[:, 9].flatten()

    for name, values in [
        ("Image", shap_image),
        ("VMAX", shap_VMAX),
        ("PER", shap_PER),
        ("POT", shap_POT),
        ("NOHC", shap_NOHC),
        ("SHDC", shap_SHDC),
        ("ICDA", shap_ICDA),
        ("D200", shap_D200),
        ("TPW", shap_TPW),
        ("PC2", shap_PC2),
        ("SDBT", shap_SDBT),
    ]:
        print(name)
        print(f"Mean: {np.mean(values)}")
        print(f"Median: {np.median(values)}")
        print(f"Max: {np.max(values)}")
        print(f"Min: {np.min(values)}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    main()
