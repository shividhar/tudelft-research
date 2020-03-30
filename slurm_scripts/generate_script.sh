#!/usr/bin/env bash

SLURM_TEMPLATE="/home/sdhar/tudelft-research/slurm_scripts/job_template.job.jinja2"
DISABLE_IB=0
DISABLE_P2P=0
NODE_LIST="node205"
GPUS_PER_NODE=1
PYTHON_SCRIPT="/home/sdhar/tudelft-research/tensorflow/tensorflow_synthetic_benchmark.py"
python3 ./tudelft-research/job_script_generator.py \
--jinja_template=$SLURM_TEMPLATE \
--disable_ib=$DISABLE_IB \
--nodes=$NODE_LIST\
--gpus_per_node=$GPUS_PER_NODE \
--script_path=" --batch-size=32" > imagenet.job
