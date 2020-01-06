# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
module load gcc
module load slurm

# Shivansh settings
module load python/3.5.2

module load cuda10.0/blas/10.0.130
module load cuda10.0/fft/10.0.130
module load cuda10.0/nsight/10.0.130
module load cuda10.0/profiler/10.0.130
module load cuda10.0/toolkit/10.0.130

module load cuDNN/cuda90/7.1

module load openmpi/cuda/64/3.1.1

export PY_USER_BIN=$(python3 -c 'import site; print(site.USER_BASE + "/bin")')
export PATH=$PY_USER_BIN:$PATH


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cm/shared/package/nccl/cuda90/nccl_2.1.2-1+cuda9.0_x86_64/lib