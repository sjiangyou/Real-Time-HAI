#!/bin/bash
# Run this script from the repository root after following the setup
# instructions in the main repository README.md.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# Prepare datasets
Rscript HRICNN/src/Model_Prep.R

# Train models
mkdir -p HRICNN/Results
python HRICNN/src/RI_TC_Prediction.py --save-model "$REPO_ROOT/HRICNN/Models/rain_ships_model.pt" --loss-csv-path HRICNN/Results/rain_ships_hyperparameter_validation_losses.csv --output-csv-path HRICNN/Results/rain_ships_model_output.csv --architecture-output-path "$REPO_ROOT/HRICNN/Results/Architectures.txt" --overwrite-architecture-output
python HRICNN/src/RI_TC_Prediction.py --image-only --save-model "$REPO_ROOT/HRICNN/Models/rain_model.pt" --loss-csv-path HRICNN/Results/rain_hyperparameter_validation_losses.csv --output-csv-path HRICNN/Results/rain_model_output.csv --architecture-output-path "$REPO_ROOT/HRICNN/Results/Architectures.txt"
Rscript HRICNN/src/SHIPS.R

# Compute SHAP values
python HRICNN/src/SHAP_Analysis.py "$REPO_ROOT/HRICNN/Models/RAIN+SHIPS.pt" --output "$REPO_ROOT/HRICNN/Results/rain_ships_shap_values.npz" --summary-output "$REPO_ROOT/HRICNN/Results/SHAP_VALUES.txt" --overwrite-summary-output --max-background 25
python HRICNN/src/SHAP_Analysis.py "$REPO_ROOT/HRICNN/Models/RAIN.pt" --output "$REPO_ROOT/HRICNN/Results/rain_shap_values.npz" --summary-output "$REPO_ROOT/HRICNN/Results/SHAP_VALUES.txt" --max-background 25

# Generate figures and statistics
Rscript HRICNN/src/Analysis.R
Rscript HRICNN/src/Plots.R
