
slurm_template = "/home/sdhar/tudelft-research/slurm_scripts/job_template.job.jinja2"
disable_ib = 0
disable_p2p = 0
node_list = "node205"
gpus_per_node = 1
python_script = "/home/sdhar/tudelft-research/tensorflow/tensorflow_synthetic_benchmark.py"

python3 ./tudelft-research/job_script_generator.py \
--jinja_template=$SLURM_TEMPLATE \
--disable_ib=$DISABLE_IB \
--nodes=$NODE_LIST\
--gpus_per_node=$GPUS_PER_NODE \
--script_path=" --batch-size=32" > imagenet.job
