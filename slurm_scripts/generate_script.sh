#!/usr/bin/env bash

SLURM_TEMPLATE="/home/sdhar/tudelft-research/slurm_scripts/job_template.job.jinja2"
DISABLE_IB=0
DISABLE_P2P=0
NODE_LIST="node206"
GPUS_PER_NODE=4
PYTHON_SCRIPT="/home/sdhar/tudelft-research/tensorflow/tensorflow_synthetic_benchmark.py"
BATCH_SIZE=32

python3 ./job_script_generator.py \
--jinja_template="$SLURM_TEMPLATE" \
--disable_ib=$DISABLE_IB \
--nodes=$NODE_LIST \
--gpus_per_node=$GPUS_PER_NODE \
--script_path="$PYTHON_SCRIPT --batch-size=$BATCH_SIZE" > TRAINING.job
