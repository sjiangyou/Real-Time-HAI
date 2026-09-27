import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def define_models():
    img_input = nn.Sequential(
        nn.Conv2d(1, 32, 5, padding=2, stride=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 5, padding=2, stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, 5, padding=2, stride=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(128 * 6 * 6, 256),
        nn.Linear(256, 170),
    )
    img_input2 = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1, stride=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1, stride=2),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 128, 3, padding=1, stride=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(128 * 6 * 6, 256),
        nn.Linear(256, 170),
    )
    img_input3 = nn.Sequential(
        nn.Conv2d(1, 32, (5, 3), padding=(2, 1)),
        nn.Conv2d(32, 32, (3, 5), padding=(1, 2)),
        nn.Conv2d(32, 32, 3, padding=1, stride=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1, stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1, stride=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(128 * 6 * 6, 256),
        nn.Linear(256, 170),
    )
    img_input4 = nn.Sequential(
        nn.Conv2d(1, 16, 4, padding=2, stride=2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, padding=2, stride=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, padding=2, stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, 4, padding=2, stride=2),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(128 * 3 * 3, 256),
        nn.Linear(256, 170),
    )
    return [
        img_input.to(DEVICE),
        img_input2.to(DEVICE),
        img_input3.to(DEVICE),
        img_input4.to(DEVICE),
    ]


def main():
    os.chdir(Path(__file__).parent.parent)
    train_data = process_train_data()
    validation_data = process_validation_data()
    test_data = process_test_data()
    models = define_models()
    train_models(models, train_data, validation_data, test_data)


def _process_data(path):
    data = pd.read_csv(path)[["GIS_ID", "VMAX", "VMAX_N06", "VMAX_N12"]]
    images, vmax, labels = [], [], []
    for i, file in enumerate(data.GIS_ID):
        filename = f"IMERG_CSV/{file}.csv"
        try:
            image = pd.read_csv(filename, header=None)
            if image.shape != (121, 121):
                continue
            images.append(np.asarray(image.iloc[40:81, 40:81]))
            labels.append(data.VMAX.iloc[i])
            vmax.append(np.array([data.VMAX_N06.iloc[i], data.VMAX_N12.iloc[i]]))
        except Exception as error:
            print(f"Error processing {filename}: {error}")
    return (
        np.asarray(images).reshape(-1, 41, 41, 1).astype("float32"),
        np.asarray(vmax).reshape(-1, 2).astype("float32"),
        np.asarray(labels),
    )


def process_train_data():
    print("Processing training data...")
    data = _process_data("IMERG/DEV/ALL_TRAIN_DATA_RESAMPLE.csv")
    print(f"Training data processed with {len(data[0])} images")
    return data


def process_validation_data():
    print("Processing validation data...")
    data = _process_data("IMERG/DEV/ALL_VALIDATION_DATA.csv")
    print(
        f"There are {len(data[0])} images, {len(data[1])} intensity values, and {len(data[2])} labels."
    )
    return data


def process_test_data():
    print("Processing test data...")
    data = _process_data("IMERG/DEV/ALL_TEST_DATA.csv")
    print(
        f"There are {len(data[0])} images, {len(data[1])} intensity values, and {len(data[2])} labels."
    )
    return data


def train_models(models, train_data, validation_data, test_data):
    os.makedirs("MODELS", exist_ok=True)
    os.makedirs("OUTPUT", exist_ok=True)
    images = torch.from_numpy(train_data[0]).to(DEVICE)
    labels = torch.from_numpy(train_data[2].astype("float32")).to(DEVICE)
    for bsize in [1, 2, 4, 8, 16]:
        for number, model in enumerate(models):
            model.to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters())
            loss_function = nn.L1Loss()
            print(f"Training model {number + 1} with batch size {bsize}.")
            for epoch in range(50):
                model.train()
                permutation = torch.randperm(len(images), device=DEVICE)
                for start in range(0, len(images), bsize):
                    indices = permutation[start : start + bsize]
                    optimizer.zero_grad()
                    loss = loss_function(model(images[indices]), labels[indices])
                    loss.backward()
                    optimizer.step()
                torch.save(
                    model.state_dict(),
                    f"MODELS/MODEL{number + 1}_EPOCHS{epoch + 1}_BATCH{bsize}.pt",
                )


if __name__ == "__main__":
    main()
