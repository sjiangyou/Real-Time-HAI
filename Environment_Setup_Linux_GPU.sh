#!/bin/bash

python -m venv .venv
source bin/activate
pip install --upgrade pip
pip install -r requirements_Linux.txt

# Locate the virtual environment directory dynamically
VIRTUAL_ENV_PATH=$(python -c "import sys; print(sys.prefix)")

# Figure out the python version
# This is needed to construct the correct path to the NVIDIA libraries in the virtual environment
VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# Manually point TensorFlow to the isolated pip NVIDIA directories
export LD_LIBRARY_PATH=$VIRTUAL_ENV_PATH/lib/python$VERSION/site-packages/nvidia/cuda_cu_device/lib:$VIRTUAL_ENV_PATH/lib/python$VERSION/site-packages/nvidia/cuda_runtime/lib:$VIRTUAL_ENV_PATH/lib/python$VERSION/site-packages/nvidia/cublas/lib:$VIRTUAL_ENV_PATH/lib/python$VERSION/site-packages/nvidia/cudnn/lib:$VIRTUAL_ENV_PATH/lib/python$VERSION/site-packages/nvidia/cufft/lib:$VIRTUAL_ENV_PATH/lib/python$VERSION/site-packages/nvidia/curand/lib:$VIRTUAL_ENV_PATH/lib/python$VERSION/site-packages/nvidia/cusolver/lib:$VIRTUAL_ENV_PATH/lib/python$VERSION/site-packages/nvidia/cusparse/lib:$VIRTUAL_ENV_PATH/lib/python$VERSION/site-packages/nvidia/nccl/lib:$VIRTUAL_ENV_PATH/lib/python$VERSION/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
