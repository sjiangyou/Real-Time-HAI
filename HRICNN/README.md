# Hurricane Rapid Intensification CNN (HRICNN)

This directory contains the PyTorch implementation of the Hurricane Rapid
Intensification Convolutional Neural Network (HRICNN). The model predicts
whether a tropical cyclone will rapidly intensify (`RI = 1`) from a 61 × 61
IMERG rainfall image and, by default, ten SHIPS environmental variables.

## Data preparation

Run the experimentation script from the repository root. First create the
shared Python and R environments with `source Environment_Setup.sh`; this
restores the R packages recorded in the root `renv.lock`. Before training,
make sure the following inputs are available:

- `BRTK_2000to2019_IMERG_SHIPS-RII.csv`
- `IMERG_CSV/<GIS_ID>.csv` for each storm observation

Run the R preprocessing script to remove incomplete records, calculate the RI
label, and create the model splits:

```bash
Rscript HRICNN/src/Model_Prep.R
```

The script writes basin-specific files to `IMERG/Model_Data/`. The training
split contains observations before 2016, the validation split contains 2016–
2017 observations, and the test split contains observations from 2016 onward.
The prepared training and validation splits are pooled for hyperparameter
selection; the test split remains untouched.

The Python loader currently trains on the Atlantic files:
`ATL_train.csv`, `ATL_val.csv`, and `ATL_test.csv`.

The commands below are also collected in `HRICNN/Experiment.sh`, which runs
both the SHIPS-enabled and image-only experiments:

```bash
./HRICNN/Experiment.sh
```

## Train and evaluate

From the repository root, run:

```bash
python HRICNN/src/RI_TC_Prediction.py
```

Training is deterministic with seed 42. The script evaluates three model
architectures, batch sizes 1, 2, 4, 8, and 16, and 1–100 epochs using focal
loss. Hyperparameters are selected with staged leave-one-year-out cross-validation
over the pooled training and validation splits. All architecture/batch
combinations are evaluated through epoch 10, then only the best batch size for
each architecture continues through the remaining epoch candidates. Each year
contributes equally to the mean validation loss. The selected architecture is
retrained on all pooled training and validation data, then evaluated on the
untouched test set.
Validation results are written to the path supplied with `--loss_csv_path`,
with one row per candidate and columns `Model`, `Epoch`, `Batch`, and `Loss`.
Pass `--output-csv-path` to save the valid test observations together with a
`Prediction` column containing the selected model's test-set predictions. Rows
whose IMERG image is missing or has the wrong shape are omitted so that every
saved prediction remains aligned with its input row.

To train using only the satellite image, without SHIPS variables, use:

```bash
python HRICNN/src/RI_TC_Prediction.py \
  --image-only \
  --loss_csv_path HRICNN/Results/image_hyperparameter_validation_losses.csv
```

Save the selected model checkpoint with `--save-model`:

```bash
python HRICNN/src/RI_TC_Prediction.py \
  --save-model HRICNN/Models/ri_model.pt \
  --loss-csv-path HRICNN/Results/ri_hyperparameter_validation_losses.csv \
  --output-csv-path HRICNN/Results/ri_test_predictions.csv
```

The checkpoint includes the selected architecture and whether SHIPS inputs
were used, so it can be loaded by the SHAP script without separately
specifying the model class.

## SHAP analysis

Generate SHAP values by passing the saved PyTorch model path as the first
argument:

```bash
python HRICNN/src/SHAP_Analysis.py HRICNN/Models/ri_model.pt \
  --output "$PWD/HRICNN/Results/ri_shap.npz"
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
python HRICNN/src/SHAP_Analysis.py HRICNN/Models/model_state_dict.pt \
  --model-class Model1 --image-only --output "$PWD/HRICNN/Results/image_shap.npz"
```

The compressed NumPy output contains `image_shap`, `predictions`, and
`expected_value`. SHIPS-enabled models also include `ships_shap`; the ten
SHIPS feature names are printed by the script. The script also prints the
aggregate `Image` and per-feature SHIPS SHAP statistics (`Mean`, `Median`,
`Max`, and `Min`) to stdout in the same format as `SHAP_VALUES.txt`.
