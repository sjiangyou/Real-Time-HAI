# Hurricane Rapid Intensification CNN (HRICNN)

This directory contains the research code for the Hurricane Rapid
Intensification Convolutional Neural Network (HRICNN). The CNN predicts
whether a tropical cyclone will rapidly intensify (`RI = 1`) from a 61 × 61
IMERG rainfall image. The default model also uses ten SHIPS environmental
variables. The directory additionally contains an image-only CNN and a
SHIPS-only linear discriminant analysis baseline.

## Current workflow

Run commands from the repository root. Create the shared Python and R
environments first with:

```bash
source Environment_Setup.sh
```

The experiment script runs the complete intended workflow:

```bash
./HRICNN/Experiment.sh
```

It prepares the data, trains the two CNN input modes, fits the SHIPS-only
baseline, generates SHAP values and plots, and prints prediction metrics.

## Data preparation

Before preprocessing, make sure these inputs are available:

- `HRICNN/BRTK_2000to2019_IMERG_SHIPS-RII.csv`
- `HRICNN/IMERG_CSV/<GIS_ID>.csv` for each storm observation

Create the RI labels and basin-specific data files with:

```bash
Rscript HRICNN/src/Model_Prep.R
```

`Model_Prep.R` removes incomplete records, defines rapid intensification as a
24-hour maximum-wind increase of at least 30 kt, adds the storm year, and
writes the following files under `HRICNN/IMERG/`:

- `<BASIN>_train.csv`: observations before 2016
- `<BASIN>_test.csv`: observations from 2016 onward
- `<BASIN>_train_resample.csv`: class-balanced resampling of the training data

There is currently no separate validation split. Hyperparameter selection
uses leave-one-year-out cross-validation within the training split, and the
test split remains untouched until final evaluation. The Python CNN code
currently loads `HRICNN/IMERG/ATL_train.csv` and `HRICNN/IMERG/ATL_test.csv`.

## CNN training and evaluation

Run the default rainfall-plus-SHIPS CNN with:

```bash
python HRICNN/src/RI_TC_Prediction.py
```

Training is deterministic with seed 42. Three architectures, batch sizes 1,
2, 4, 8, and 16, and epoch counts from 1 through 100 are evaluated using
binary focal loss. Selection is staged: every architecture/batch-size
combination is evaluated through epoch 10, then the best batch size for each
architecture is evaluated through the remaining epoch candidates.

The selection metric is:

```text
OS = BSS + POD - FAR + PSS
```

BSS uses a uniform binary climatological probability of 0.5. POD, FAR, and
PSS use the 0.21 classification boundary. Each held-out year contributes
equally to the cross-validation mean. The selected architecture is retrained
on the complete training split and evaluated on the untouched test split.

Validation results are written to stage-specific CSV files. For example,
`--loss-csv-path results/losses.csv` creates:

- `results/losses_stage1.csv`
- `results/losses_stage2.csv`

The files contain `Model`, `Epoch`, `Batch`, `Loss`, `BSS`, `POD`, `FAR`,
`PSS`, and `OS` columns. Pass `--output-csv-path` to save valid test rows and
their predictions, and `--save-model` to save a checkpoint containing the
selected architecture and input mode.

To train the image-only CNN, use:

```bash
python HRICNN/src/RI_TC_Prediction.py \
    --image-only \
    --loss-csv-path HRICNN/Results/image_hyperparameter_validation_losses.csv
```

The experiment script saves the rainfall-plus-SHIPS model as
`HRICNN/Models/Rain_Model.pt` and the image-only model as
`HRICNN/Models/Model.pt`. Their test predictions are saved as
`rain_model_output.csv` and `model_output.csv`, respectively.

## SHIPS-only baseline

Fit the SHIPS-only linear discriminant analysis baseline with:

```bash
Rscript HRICNN/src/SHIPS.R
```

The script uses `HRICNN/IMERG/ATL_train_resample.csv` and
`HRICNN/IMERG/ATL_test.csv`, then writes
`HRICNN/Results/SHIPS_predictions.csv`.

## Prediction analysis

Calculate metrics for the rainfall-plus-SHIPS CNN, image-only CNN, and
SHIPS-only baseline with:

```bash
Rscript HRICNN/src/Analysis.R
```

The script validates that all three output files contain matching observations
and reports Brier Score, Brier Skill Score, Probability of Detection (POD),
False Alarm Ratio (FAR), and Peirce Skill Score (PSS). Brier-based metrics use
prediction probabilities directly; threshold-based metrics use the
configurable `classification_boundary` in `Analysis.R`, currently `0.21`.
It also saves the derived overall score, `OS = BSS + POD - FAR + PSS`, to
`HRICNN/Results/model_metrics.csv`.

## Hyperparameter plots

Generate plots from the saved staged validation CSV files with:

```bash
Rscript HRICNN/src/Plots.R
```

The script writes plots to `HRICNN/Results/Plots/`. Stage 1 plots summarize
the maximum OS for each batch size; stage 2 plots show OS by epoch and
highlight the best epoch. It also writes
`model_metric_comparisons.png`, a single five-panel test-metric comparison
figure arranged left-to-right as POD, FAR, PSS, BSS, and OS with a shared
y-axis. Model colors are identified by a shared legend, rather than repeated
x-axis labels. The comparison function reads model names from
`model_metrics.csv`, so additional model rows are included automatically as
long as the CSV contains those five metric columns. The
current plot script distinguishes input modes
by filename prefix, so its expected prefixes should be kept consistent with
the loss-file paths passed to `RI_TC_Prediction.py`.

## SHAP analysis

Generate SHAP values for a saved checkpoint with:

```bash
python HRICNN/src/SHAP_Analysis.py HRICNN/Models/Rain_Model.pt \
    --output HRICNN/Results/rain_shap_values.npz
```

Optional arguments include:

```text
--max-background N   Background examples per RI class (default: 100)
--max-samples N      Explain at most N test observations
```

The compressed NumPy output contains image SHAP values, predictions, and the
expected value. SHIPS-enabled checkpoints also contain SHIPS SHAP values, and
the script prints aggregate image and per-feature SHIPS statistics.

## Current implementation notes

- The CNN training loader reads data from `HRICNN/IMERG/`, while the current
  SHAP script still references `IMERG/Model_Data/` for its background and test
  data. These paths must be reconciled before the SHAP step can run unchanged
  after preprocessing.
- `Plots.R` determines the input mode from filename prefixes. Verify the
  prefixes against the files produced by `Experiment.sh` before interpreting
  the generated figures.
- Results, model checkpoints, generated plots, and SHAP archives are runtime
  outputs; the source scripts and this README document the reproducible
  workflow.
