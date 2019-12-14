#!/usr/bin/env bash
set -e

##### Script to setup a Python virtualenv with PyTorch #####

# Load Python3.5.2 to local environment because it is supported by
# PyTorch
module load python/3.5.2

python3 --version

# Install latest version of virtualenv
pip3 install --user virtualenv

# Path to locally install Python packages
PY_USER_BIN=$(python3 -c 'import site; print(site.USER_BASE + "/bin")')

# Create virtual environment with user parameter
$PY_USER_BIN/virtualenv -p python3 $1

# Go into virtualenv
source $1/bin/activate

PIP3_PATH=$(which pip3)

# #### Fixes bug with Python3.5.2 pip
sed -i 's/from pip._internal.main import main/from pip import main/g' $PIP3_PATH

# Install PyTorch
pip3 install torch==1.3.1+cu100 torchvision==0.4.2+cu100 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install horovod
