# Author: Sunny You
import os
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
EPOCHS = 6
EPOCH_CANDIDATES = range(1, 101)
BATCH_SIZE_CANDIDATES = (1, 2, 4, 8, 16)
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1


class BinaryFocalCrossEntropy(nn.Module):
    """Binary focal crossentropy for models that return sigmoid probabilities."""

    def __init__(self):
        super().__init__()
        self.gamma = 2
        self.alpha = 1 - 0.07036247334754797  # Better match our data's class imbalance

    def forward(self, predictions, targets):
        bce = F.binary_cross_entropy(predictions, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt).pow(self.gamma) * bce

        return focal_loss.mean()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _save_model_checkpoint(model, path):
    """Save the selected model and the input schema needed to restore it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_class": model.__class__.__name__,
            "use_ships": bool(model.use_ships),
            "state_dict": model.state_dict(),
        },
        path,
    )
    print(f"Saved model to {path}")


def main(image_only=False, save_model=None, loss_csv_path=None, output_csv_path=None):
    gpu_test()

    data, test_dataframe = load_data(
        include_year=True,
        return_test_dataframe=True,
    )
    (
        train_img,
        train_ships,
        train_label,
        train_year,
        test_img,
        test_ships,
        test_label,
    ) = data
    (
        validation_img,
        validation_ships,
        validation_label,
        validation_year,
    ) = load_validation_data(include_year=True)
    models = model_setup(
        train_img,
        train_ships,
        train_label,
        test_img,
        test_ships,
        test_label,
        use_ships=not image_only,
    )

    trained_model, predictions = run_models(
        *models,
        train_img,
        train_ships,
        train_label,
        train_year,
        validation_img,
        validation_ships,
        validation_label,
        validation_year,
        test_img,
        test_ships,
        test_label,
        use_ships=not image_only,
        loss_csv_path=loss_csv_path,
    )

    if output_csv_path is not None:
        test_dataframe = test_dataframe.copy()
        test_dataframe["Prediction"] = predictions
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        test_dataframe.to_csv(output_csv_path, index=False)
        print(f"Test predictions saved to {output_csv_path}")

    if save_model is not None:
        _save_model_checkpoint(trained_model, save_model)

    return trained_model


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
# Seeding
# ---------------------------------------------------------------------------


def seed_everything(seed=42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_arrays(dataframe, return_valid_indices=False):
    """Load model arrays and optionally return rows with valid image inputs."""
    images, ships_data, labels, years = [], [], [], []
    valid_indices = []
    for index, row in enumerate(dataframe.itertuples(index=False)):
        try:
            temp = pd.read_csv(f"IMERG_CSV/{row.GIS_ID}.csv", header=None)
            if temp.shape != (121, 121):
                continue
            images.append(np.asarray(temp.iloc[30:91, 30:91]))
            labels.append(row.RI)
            if hasattr(row, "Year"):
                years.append(row.Year)
            ships_data.append(
                np.asarray(
                    [
                        row.VMAX,
                        row.PER,
                        row.POT,
                        row.NOHC,
                        row.SHDC,
                        row.ICDA,
                        row.D200,
                        row.TPW,
                        row.PC2,
                        row.SDBT,
                    ]
                )
            )
            valid_indices.append(index)
        except Exception:
            pass

    X_img = np.asarray(images).reshape(-1, 61, 61, 1).astype("float32")
    arrays = (
        np.transpose(X_img, (0, 3, 1, 2)),
        np.asarray(ships_data).reshape(-1, 10).astype("float32"),
        np.asarray(labels).astype("float32"),
        np.asarray(years),
    )
    if return_valid_indices:
        return arrays, np.asarray(valid_indices, dtype=np.int64)
    return arrays


def _prepare_dataframe(path):
    dataframe = pd.read_csv(path)
    columns = [
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
    dataframe = dataframe[columns]
    dataframe.columns = [
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
    ]
    return dataframe


def load_data(include_year=False, return_test_dataframe=False):
    """Load train/test arrays and optionally the prediction-aligned test dataframe."""
    train = _load_arrays(_prepare_dataframe("IMERG/Model_Data/ATL_train.csv"))
    test_dataframe = _prepare_dataframe("IMERG/Model_Data/ATL_test.csv")
    if return_test_dataframe:
        test, valid_indices = _load_arrays(
            test_dataframe,
            return_valid_indices=True,
        )
        test_dataframe = test_dataframe.iloc[valid_indices].reset_index(drop=True)
    else:
        test = _load_arrays(test_dataframe)
    data = train[:3] + test[:3]
    if include_year:
        data = train[:3] + (train[3],) + test[:3]
    if return_test_dataframe:
        return data, test_dataframe
    return data


def load_validation_data(include_year=False):
    """Load the explicit validation split using the shared dataframe loader."""
    validation = _load_arrays(_prepare_dataframe("IMERG/Model_Data/ATL_val.csv"))
    if include_year:
        return validation
    return validation[:3]


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

    def __init__(self, use_ships=True):
        super().__init__()
        self.use_ships = use_ships

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
        self.fc1 = nn.Linear(1024 + (10 if use_ships else 0), 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, image, ships=None):
        x = self.image_layers(image)
        x = torch.flatten(x, start_dim=1)
        if self.use_ships:
            x = torch.cat((x, ships), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)

        return torch.sigmoid(x)


class Model2(nn.Module):
    """
    Keras model_2 converted to PyTorch.
    """

    def __init__(self, use_ships=True):
        super().__init__()
        self.use_ships = use_ships

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
        self.fc = nn.Linear(256 + (10 if use_ships else 0), 1)

    def forward(self, image, ships=None):
        x = self.image_layers(image)
        x = torch.flatten(x, start_dim=1)
        if self.use_ships:
            x = torch.cat((x, ships), dim=1)
        x = self.fc(x)

        return torch.sigmoid(x)


class Model3(nn.Module):
    """
    Keras model_3 converted to PyTorch.
    """

    def __init__(self, use_ships=True):
        super().__init__()
        self.use_ships = use_ships

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
        self.fc = nn.Linear(576 + (10 if use_ships else 0), 1)

    def forward(self, image, ships=None):
        x = self.image_layers(image)
        x = torch.flatten(x, start_dim=1)
        if self.use_ships:
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
    use_ships=True,
):
    new_model1 = Model1(use_ships=use_ships).to(DEVICE)
    new_model2 = Model2(use_ships=use_ships).to(DEVICE)
    new_model3 = Model3(use_ships=use_ships).to(DEVICE)

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


def _make_dataset(X_img, X_ships, y):
    dataset = TensorDataset(
        torch.from_numpy(X_img),
        torch.from_numpy(X_ships),
        torch.from_numpy(y),
    )
    return dataset


def _train_epoch(model, train_loader, criterion, optimizer):
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

        count = labels.shape[0]
        running_loss += loss.item() * count
        running_count += count

    return running_loss / running_count


def train_model(
    model,
    X_train_img,
    X_train_ships,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    seed=42,
    use_ships=None,
):
    if use_ships is not None and bool(model.use_ships) != bool(use_ships):
        raise ValueError("Model input configuration does not match use_ships.")
    dataset = _make_dataset(X_train_img, X_train_ships, y_train)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )

    criterion = BinaryFocalCrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {train_loss:.6f}")

    return model


def select_hyperparameters(
    X_train_img,
    X_train_ships,
    y_train,
    years_train,
    X_validation_img=None,
    X_validation_ships=None,
    y_validation=None,
    years_validation=None,
    epoch_candidates=EPOCH_CANDIDATES,
    batch_size_candidates=BATCH_SIZE_CANDIDATES,
    model_classes=(Model1, Model2, Model3),
    seed=42,
    use_ships=True,
    loss_csv_path=None,
):
    """Select hyperparameters with staged leave-one-year-out cross-validation.

    The prepared training and validation splits are pooled, while the test
    split is never passed to this function. Each distinct year is held out in
    turn. Year-level losses are averaged equally, so years with more samples
    do not dominate selection. All architecture/batch combinations are
    evaluated through epoch 10 first; only the best batch size for each
    architecture is evaluated beyond epoch 10.
    """
    if (
        X_validation_img is None
        or X_validation_ships is None
        or y_validation is None
        or years_train is None
        or years_validation is None
    ):
        raise ValueError("Training and validation arrays with years are required.")
    epoch_candidates = tuple(sorted(set(int(epoch) for epoch in epoch_candidates)))
    batch_size_candidates = tuple(
        sorted(set(int(size) for size in batch_size_candidates))
    )
    if not epoch_candidates or not batch_size_candidates or not model_classes:
        raise ValueError("Candidate architectures and hyperparameters cannot be empty.")
    max_epochs = max(epoch_candidates)
    loss_csv_path = Path(loss_csv_path or "hyperparameter_validation_losses.csv")
    loss_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["Model", "Epoch", "Batch", "Loss"]).to_csv(
        loss_csv_path, index=False
    )
    train_img = np.concatenate((X_train_img, X_validation_img), axis=0)
    train_ships = np.concatenate((X_train_ships, X_validation_ships), axis=0)
    train_labels = np.concatenate((y_train, y_validation), axis=0)
    years = np.concatenate((years_train, years_validation), axis=0)
    unique_years = np.unique(years)
    if unique_years.size < 2:
        raise ValueError("At least two distinct years are required for validation.")
    if not (len(train_img) == len(train_ships) == len(train_labels) == len(years)):
        raise ValueError(
            "Training, labels, and years must contain the same number of samples."
        )

    scores = np.full(
        (len(model_classes), len(batch_size_candidates), max_epochs),
        np.nan,
        dtype=np.float64,
    )
    criterion = BinaryFocalCrossEntropy()

    def evaluate_configuration(
        model_index, model_class, batch_index, batch_size, epochs_to_run
    ):
        """Train one architecture/batch configuration across all year folds."""
        fold_losses = np.full((len(unique_years), max_epochs), np.nan)
        for fold_index, held_out_year in enumerate(unique_years):
            train_indices = np.flatnonzero(years != held_out_year)
            validation_indices = np.flatnonzero(years == held_out_year)
            model_seed = seed + model_index * 1000 + batch_index * 100 + fold_index
            torch.manual_seed(model_seed)
            model = model_class(use_ships=use_ships).to(DEVICE)
            train_loader = DataLoader(
                _make_dataset(
                    train_img[train_indices],
                    train_ships[train_indices],
                    train_labels[train_indices],
                ),
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(model_seed),
            )
            validation_loader = DataLoader(
                _make_dataset(
                    train_img[validation_indices],
                    train_ships[validation_indices],
                    train_labels[validation_indices],
                ),
                batch_size=max(batch_size_candidates),
                shuffle=False,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            for epoch in range(epochs_to_run):
                _train_epoch(model, train_loader, criterion, optimizer)
                fold_losses[fold_index, epoch] = evaluate_model(
                    model, validation_loader, criterion
                )[0]
            print(
                f"{model_class.__name__} | batch size={batch_size} | "
                f"held-out year={held_out_year} complete",
                flush=True,
            )
        return np.mean(fold_losses, axis=0)

    stage_one_epochs = min(10, max_epochs)
    stage_one_candidates = tuple(
        epoch for epoch in epoch_candidates if epoch <= stage_one_epochs
    )
    stage_two_candidates = tuple(
        epoch for epoch in epoch_candidates if epoch > stage_one_epochs
    )

    # Stage one evaluates every architecture/batch pair only through epoch 10.
    # This identifies one promising batch size for each architecture.
    stage_one_best_batches = {}
    for model_index, model_class in enumerate(model_classes):
        best_batch_index = None
        best_stage_one_loss = np.inf
        for batch_index, batch_size in enumerate(batch_size_candidates):
            print(
                f"Starting hyperparameter test: architecture={model_class.__name__}, "
                f"batch size={batch_size}, epochs=1-{stage_one_epochs}, "
                f"years={len(unique_years)}",
                flush=True,
            )
            mean_losses = evaluate_configuration(
                model_index,
                model_class,
                batch_index,
                batch_size,
                stage_one_epochs,
            )
            for epoch in stage_one_candidates:
                validation_loss = mean_losses[epoch - 1]
                scores[model_index, batch_index, epoch - 1] = validation_loss
                pd.DataFrame(
                    [
                        {
                            "Model": model_class.__name__,
                            "Epoch": epoch,
                            "Batch": batch_size,
                            "Loss": validation_loss,
                        }
                    ]
                ).to_csv(loss_csv_path, mode="a", header=False, index=False)
            print(
                f"Validation architecture {model_class.__name__}, "
                f"batch size {batch_size} complete; "
                f"stage-one loss={np.min(mean_losses[:stage_one_epochs]):.6f}"
            )
            stage_one_loss = np.min(mean_losses[:stage_one_epochs])
            if stage_one_loss < best_stage_one_loss:
                best_stage_one_loss = stage_one_loss
                best_batch_index = batch_index
        stage_one_best_batches[model_index] = best_batch_index
        print(
            f"Stage one selected {model_class.__name__} batch size "
            f"{batch_size_candidates[best_batch_index]}"
        )

    # Stage two searches epoch counts only for the best batch size per
    # architecture. The selected configurations are retrained from epoch one
    # so their later-epoch losses remain directly comparable.
    for model_index, model_class in enumerate(model_classes):
        batch_index = stage_one_best_batches[model_index]
        batch_size = batch_size_candidates[batch_index]
        if not stage_two_candidates:
            continue
        print(
            f"Starting stage two: architecture={model_class.__name__}, "
            f"batch size={batch_size}, epochs=1-{max_epochs}, "
            f"years={len(unique_years)}",
            flush=True,
        )
        mean_losses = evaluate_configuration(
            model_index,
            model_class,
            batch_index,
            batch_size,
            max_epochs,
        )
        for epoch in stage_two_candidates:
            validation_loss = mean_losses[epoch - 1]
            scores[model_index, batch_index, epoch - 1] = validation_loss
            pd.DataFrame(
                [
                    {
                        "Model": model_class.__name__,
                        "Epoch": epoch,
                        "Batch": batch_size,
                        "Loss": validation_loss,
                    }
                ]
            ).to_csv(loss_csv_path, mode="a", header=False, index=False)
        print(
            f"Stage two complete: {model_class.__name__}; " f"batch size={batch_size}"
        )

    print(f"Validation losses saved to {loss_csv_path}")

    best_model_index, best_batch_index, best_epoch_index = np.unravel_index(
        np.nanargmin(scores), scores.shape
    )
    best_model_class = model_classes[best_model_index]
    best_batch_size = batch_size_candidates[best_batch_index]
    best_epochs = best_epoch_index + 1
    print(
        f"Selected architecture: {best_model_class.__name__}; "
        f"epochs: {best_epochs}; batch size: {best_batch_size}; "
        f"validation focal loss: "
        f"{scores[best_model_index, best_batch_index, best_epoch_index]:.6f}"
    )
    return best_model_class, best_epochs, best_batch_size


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
    years_train,
    X_validation_img,
    X_validation_ships,
    y_validation,
    years_validation,
    X_test_img,
    X_test_ships,
    y_test,
    use_ships=True,
    loss_csv_path=None,
):
    best_model_class, best_epochs, best_batch_size = select_hyperparameters(
        X_train_img,
        X_train_ships,
        y_train,
        years_train,
        X_validation_img,
        X_validation_ships,
        y_validation,
        years_validation,
        use_ships=use_ships,
        loss_csv_path=loss_csv_path,
    )

    # Selection models are discarded; fit the winning architecture on all
    # pooled train/validation samples before evaluating on the untouched test set.
    new_model1 = best_model_class(use_ships=use_ships).to(DEVICE)
    new_model1 = train_model(
        new_model1,
        np.concatenate((X_train_img, X_validation_img), axis=0),
        np.concatenate((X_train_ships, X_validation_ships), axis=0),
        np.concatenate((y_train, y_validation), axis=0),
        epochs=best_epochs,
        batch_size=best_batch_size,
        use_ships=use_ships,
    )

    test_dataset = TensorDataset(
        torch.from_numpy(X_test_img),
        torch.from_numpy(X_test_ships),
        torch.from_numpy(y_test),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=best_batch_size,
        shuffle=False,
    )

    criterion = BinaryFocalCrossEntropy()

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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate the HRICNN rapid-intensification models."
    )
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Ignore all SHIPS variables and train models from image data only.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        help="Save the selected model checkpoint for later SHAP analysis.",
    )
    parser.add_argument(
        "--loss-csv-path",
        type=Path,
        default=Path("rain_hyperparameter_validation_losses.csv"),
        help="Path to save hyperparameter tuning losses.",
    )
    parser.add_argument(
        "--output-csv-path",
        type=Path,
        help="Path to save the test dataframe with model predictions.",
    )
    args = parser.parse_args()
    # Resolve output paths before changing directories so relative paths are
    # always relative to the directory from which the command was launched.
    if args.save_model is not None:
        args.save_model = args.save_model.resolve()
    args.loss_csv_path = args.loss_csv_path.resolve()
    if args.output_csv_path is not None:
        args.output_csv_path = args.output_csv_path.resolve()
    os.chdir(Path(__file__).parent.parent)
    seed_everything(seed=42)
    main(
        image_only=args.image_only,
        save_model=args.save_model,
        loss_csv_path=args.loss_csv_path,
        output_csv_path=args.output_csv_path,
    )
