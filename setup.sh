#!/usr/bin/env bash
set -e

##### Script to setup a Python virtualenv with PyTorch #####

# Check to see if the virtualenv path has been specified
[[ -z "$1" ]] && { echo "Please pass in the desired path for the virtualenv." ; exit 1; }

# Preserve system's bashrc and replace with project's
mv ~/.bashrc ~/bashrc
ln -s ./bashrc ~/.bashrc
source ~/.bashrc

# Load Python3.5.2 to local environment because it is supported by
# PyTorch
module load python/3.5.2

module load cuda10.0/blas/10.0.130
module load cuda10.0/fft/10.0.130
module load cuda10.0/nsight/10.0.130
module load cuda10.0/profiler/10.0.130
module load cuda10.0/toolkit/10.0.130

module load cuDNN/cuda90/7.1

module load openmpi/cuda/64/3.1.1

python3 --version

# Install latest version of virtualenv
pip3 install --user virtualenv

# Install Jinja to generate SLURM Scripts
pip3 install --user jinja2

# Path to locally install Python packages
PY_USER_BIN=$(python3 -c 'import site; print(site.USER_BASE + "/bin")')

# Create virtual environment with user parameter
# $PY_USER_BIN/virtualenv -p python3 $1

# Go into virtualenv
source $1/bin/activate

CC=$(which gcc)
CXX=$(which g++)
PIP3_PATH=$(which pip3)

# #### Fixes bug with Python3.5.2 pip
# sed -i 's/from pip._internal.main import main/from pip import main/g' $PIP3_PATH

# Install PyTorch (arguments specifically to system's CUDA version)
pip3 install torch==1.3.1+cu100 torchvision==0.4.2+cu100 -f https://download.pytorch.org/whl/torch_stable.html

# This will install tensorflow2
pip3 install tensorflow

# Install Horovod with Pytorch and Tensorflow support
# Step 5 of https://github.com/horovod/horovod/blob/master/docs/gpus.rst
# If install is failing, add CC and CXX to intall line
HOROVOD_NCCL_HOME=/cm/shared/package/nccl/cuda90/nccl_2.1.2-1+cuda9.0_x86_64 HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
pip3 install --no-cache-dir horovod

PYTHONPATH=$1/lib/python3.5/site-packages
