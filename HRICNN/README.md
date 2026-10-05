# Hurricane Rapid Intensification CNN (HRICNN)

This directory contains the PyTorch implementation of the Hurricane Rapid
Intensification Convolutional Neural Network (HRICNN). The model predicts
whether a tropical cyclone will rapidly intensify (`RI = 1`) from a 61 × 61
IMERG rainfall image and, by default, ten SHIPS environmental variables.

## Data preparation

The scripts expect to be run from this directory. Before training, make sure
the following inputs are available:

- `BRTK_2000to2019_IMERG_SHIPS-RII.csv`
- `IMERG_CSV/<GIS_ID>.csv` for each storm observation

Run the R preprocessing script to remove incomplete records, calculate the RI
label, and create the model splits:

```bash
Rscript src/Model_Prep.R
```

The script writes basin-specific files to `IMERG/Model_Data/`. The training
split contains observations before 2016, the validation split contains 2016–
2017 observations, and the test split contains observations from 2016 onward.
The validation split is used for hyperparameter selection before final test
evaluation.

The Python loader currently trains on the Atlantic files:
`ATL_train.csv`, `ATL_val.csv`, and `ATL_test.csv`.

## Train and evaluate

From `HRICNN/`, run:

```bash
python src/RI_TC_Prediction.py
```

Training is deterministic with seed 42. The script evaluates three model
architectures, batch sizes 1, 2, 4, 8, and 16, and 1–100 epochs using focal
loss on the validation split. It selects the combination with the lowest
validation loss, retrains that architecture on the training split, and reports
test loss, MAE, and MSE. Validation results are written to
`rain_hyperparameter_validation_losses.csv`.

To train using only the satellite image, without SHIPS variables, use:

```bash
python src/RI_TC_Prediction.py --image-only
```

Save the selected model checkpoint with `--save-model`:

```bash
python src/RI_TC_Prediction.py --save-model models/ri_model.pt
```

The checkpoint includes the selected architecture and whether SHIPS inputs
were used, so it can be loaded by the SHAP script without separately
specifying the model class.

## SHAP analysis

Generate SHAP values by passing the saved PyTorch model path as the first
argument:

```bash
python src/SHAP_Analysis.py models/ri_model.pt --output results/ri_shap.npz
```

The command uses a balanced background sample from the training data and
explains the test observations. Optional arguments are:

```text
--max-background N   Background examples per RI class (default: 100)
--max-samples N      Explain at most N test observations
```

For a raw `state_dict`, specify its architecture with `--model-class`
(`Model1`, `Model2`, or `Model3`). Add `--image-only` when that raw state dict
was trained without SHIPS variables:

```bash
python src/SHAP_Analysis.py models/model_state_dict.pt \
  --model-class Model1 --image-only --output results/image_shap.npz
```

The compressed NumPy output contains `image_shap`, `predictions`, and
`expected_value`. SHIPS-enabled models also include `ships_shap`; the ten
SHIPS feature names are printed by the script. The script also prints the
aggregate `Image` and per-feature SHIPS SHAP statistics (`Mean`, `Median`,
`Max`, and `Min`) to stdout in the same format as `SHAP_VALUES.txt`.
