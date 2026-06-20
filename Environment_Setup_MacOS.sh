#!/bin/bash

python -m venv .venv
source bin/activate
pip install --upgrade pip
pip install -r requirements_MacOS.txt
