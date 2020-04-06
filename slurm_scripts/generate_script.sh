#!/usr/bin/env bash

SLURM_TEMPLATE="/home/sdhar/tudelft-research/slurm_scripts/job_template.job.jinja2"
DISABLE_IB=0
DISABLE_P2P=0
NODE_LIST="node001,node002"
#NODE_LIST="node001,node002,node003,node004,node005,node006,node007,node024"
GPUS_PER_NODE=1
PYTHON_SCRIPT="/home/sdhar/tudelft-research/pytorch/pytorch_synthetic_benchmark.py"
BATCH_SIZE=32

python3 ./job_script_generator.py \
--jinja_template="$SLURM_TEMPLATE" \
--nodes=$NODE_LIST \
--gpus_per_node=$GPUS_PER_NODE \
--disable_ib=$DISABLE_IB \
--disable_p2p=$DISABLE_P2P \
--script_path="$PYTHON_SCRIPT --batch-size=$BATCH_SIZE --disable_ib=$DISABLE_IB --disable_p2p=$DISABLE_P2P" > TRAINING.job
