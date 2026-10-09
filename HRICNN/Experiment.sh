#!/bin/bash
# Run this script from the repository root after following the setup
# instructions in the main repository README.md.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# Prepare datasets
Rscript HRICNN/src/Model_Prep.R

# Train models
mkdir -p HRICNN/Results
python HRICNN/src/RI_TC_Prediction.py --save-model "$REPO_ROOT/HRICNN/Models/Rain_Model.pt" --loss-csv-path HRICNN/Results/rain_hyperparameter_validation_losses.csv
python HRICNN/src/RI_TC_Prediction.py --image-only --save-model "$REPO_ROOT/HRICNN/Models/Model.pt" --loss-csv-path HRICNN/Results/hyperparameter_validation_losses.csv

# Compute SHAP values
python HRICNN/src/SHAP_Analysis.py "$REPO_ROOT/HRICNN/Models/Rain_Model.pt" --output "$REPO_ROOT/HRICNN/Results/rain_shap_values.npz" --max-background 25
python HRICNN/src/SHAP_Analysis.py "$REPO_ROOT/HRICNN/Models/Model.pt" --output "$REPO_ROOT/HRICNN/Results/shap_values.npz" --max-background 25
