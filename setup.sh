#!/usr/bin/env bash

# Script to setup a python virtualenv with PyTorch

# Load Python3.6 to local environment because it is supported by
# PyTorch
module load python/3.6.0

# Install latest version of virtualenv
pip3 install --user virtualenv

# Path to locally install Python packages
PY_USER_BIN=$(python3 -c 'import site; print(site.USER_BASE + "/bin")')

# Create virtual environment with user parameter
$PY_USER_BIN/virtualenv -p python3 $1

# Go into virtualenv
source $1/bin/activate

# Install PyTorch
pip3 install torch torchvision
