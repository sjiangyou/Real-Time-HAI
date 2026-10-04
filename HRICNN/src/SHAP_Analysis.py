"""Generate SHAP values for a saved HRICNN PyTorch model.

Run from ``HRICNN`` (or let this script change to that directory)::

    python src/SHAP_Analysis.py path/to/model.pt --output shap_values.npz

The output is a compressed NumPy archive containing ``image_shap`` and,
when applicable, ``ships_shap``.  The archive also contains predictions and
the explainer expected value.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch

from RI_TC_Prediction import Model1, Model2, Model3, _load_arrays, _prepare_dataframe

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHIPS_NAMES = (
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
)


def _load_model(model_path, model_class=None, image_only=False):
    """Load a full module, checkpoint, or raw state dict saved by PyTorch."""
    try:
        saved = torch.load(model_path, map_location=DEVICE, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        saved = torch.load(model_path, map_location=DEVICE)
    if isinstance(saved, torch.nn.Module):
        model = saved
        use_ships = getattr(model, "use_ships", not image_only)
    else:
        metadata = saved if isinstance(saved, dict) else {}
        state_dict = metadata.get("model_state_dict", metadata.get("state_dict"))
        if state_dict is None and all(isinstance(key, str) for key in metadata):
            state_dict = metadata
        if state_dict is None:
            raise ValueError("The model file does not contain a PyTorch state dict.")

        class_name = model_class or metadata.get("model_class", "Model1")
        classes = {"Model1": Model1, "Model2": Model2, "Model3": Model3}
        if class_name not in classes:
            raise ValueError(
                f"Unknown model class {class_name!r}; use Model1, Model2, or Model3."
            )
        use_ships = metadata.get("use_ships", not image_only)
        model = classes[class_name](use_ships=use_ships)
        model.load_state_dict(state_dict)

    return model.to(DEVICE).eval(), bool(use_ships)


def _background_arrays(max_background):
    dataframe = _prepare_dataframe("IMERG/Model_Data/ATL_train.csv")
    false_rows = dataframe[dataframe["RI"] == 0].sample(
        min(max_background, (dataframe["RI"] == 0).sum()), random_state=42
    )
    true_rows = dataframe[dataframe["RI"] == 1].sample(
        min(max_background, (dataframe["RI"] == 1).sum()), random_state=42
    )
    background = pd.concat([false_rows, true_rows]).reset_index(drop=True)
    return _load_arrays(background)[:2]


def _normalise_shap_values(values, use_ships):
    if not use_ships:
        if isinstance(values, list):
            values = values[0]
        return np.asarray(values), None
    if isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
        values = values[0]
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError("Unexpected SHAP output format for the model inputs.")
    return np.asarray(values[0]), np.asarray(values[1])


def _print_summary(image_shap, ships_shap):
    """Print the aggregate SHAP report used by ``SHAP_VALUES.txt``."""
    summaries = [
        ("Image", image_shap.reshape(len(image_shap), -1).sum(axis=1))
    ]
    if ships_shap is not None:
        ships_shap = ships_shap.reshape(len(ships_shap), -1)
        summaries.extend(
            (name, ships_shap[:, index])
            for index, name in enumerate(SHIPS_NAMES)
        )

    for name, values in summaries:
        print(name)
        print(f"Mean: {np.mean(values)}")
        print(f"Median: {np.median(values)}")
        print(f"Max: {np.max(values)}")
        print(f"Min: {np.min(values)}")
        print()


def generate_shap_values(
    model_path,
    output_path,
    model_class=None,
    image_only=False,
    max_background=100,
    max_samples=None,
):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"PyTorch model file not found: {model_path}")
    if max_background < 1:
        raise ValueError("max_background must be at least 1.")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be at least 1 when provided.")

    model, use_ships = _load_model(model_path, model_class, image_only)
    test_img, test_ships, _, _ = _load_arrays(
        _prepare_dataframe("IMERG/Model_Data/ATL_test.csv")
    )
    if max_samples is not None:
        test_img, test_ships = test_img[:max_samples], test_ships[:max_samples]
    background_img, background_ships = _background_arrays(max_background)

    background_img_tensor = torch.from_numpy(background_img).to(DEVICE)
    test_img_tensor = torch.from_numpy(test_img).to(DEVICE)
    if use_ships:
        background = [
            background_img_tensor,
            torch.from_numpy(background_ships).to(DEVICE),
        ]
        inputs = [test_img_tensor, torch.from_numpy(test_ships).to(DEVICE)]
    else:
        background, inputs = background_img_tensor, test_img_tensor

    explainer = shap.DeepExplainer(model, background)
    values = explainer.shap_values(inputs, check_additivity=False)
    image_shap, ships_shap = _normalise_shap_values(values, use_ships)
    if image_shap.ndim == 5 and image_shap.shape[-1] == 1:
        image_shap = image_shap[..., 0]
    if ships_shap is not None and ships_shap.ndim == 3 and ships_shap.shape[-1] == 1:
        ships_shap = ships_shap[..., 0]

    _print_summary(image_shap, ships_shap)

    with torch.no_grad():
        predictions = (
            model(test_img_tensor, inputs[1] if use_ships else None)
            .cpu()
            .numpy()
            .reshape(-1)
        )
    expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "image_shap": image_shap,
        "predictions": predictions,
        "expected_value": np.asarray(expected),
        "use_ships": np.asarray(use_ships),
    }
    if ships_shap is not None:
        arrays["ships_shap"] = ships_shap
    np.savez_compressed(output_path, **arrays)
    print(f"Saved SHAP values to {output_path}")
    print(f"Image SHAP shape: {image_shap.shape}")
    if ships_shap is not None:
        print(f"SHIPS SHAP shape: {ships_shap.shape}")
        print("SHIPS features:", json.dumps(SHIPS_NAMES))
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to the saved PyTorch module, checkpoint, or state dict.",
    )
    parser.add_argument("--output", type=Path, default=Path("shap_values.npz"))
    parser.add_argument(
        "--model-class",
        choices=("Model1", "Model2", "Model3"),
        help="Architecture for a raw state dict without metadata.",
    )
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Treat a raw state dict as an image-only model.",
    )
    parser.add_argument(
        "--max-background",
        type=int,
        default=100,
        help="Number of background examples per RI class (default: 100).",
    )
    parser.add_argument(
        "--max-samples", type=int, help="Limit the number of test examples explained."
    )
    args = parser.parse_args()
    model_path = args.model_path.resolve()
    os.chdir(Path(__file__).resolve().parent.parent)
    generate_shap_values(
        model_path,
        args.output,
        args.model_class,
        args.image_only,
        args.max_background,
        args.max_samples,
    )


if __name__ == "__main__":
    main()
