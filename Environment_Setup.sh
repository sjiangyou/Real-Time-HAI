#!/bin/bash

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Restore the project-local R library recorded in renv.lock.
Rscript --vanilla -e 'source("renv/activate.R"); renv::restore(prompt = FALSE)'
