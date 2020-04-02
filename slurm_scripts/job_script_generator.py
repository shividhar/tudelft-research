import argparse
from jinja2 import Template

# Defaults
slurm_template = "/home/sdhar/tudelft-research/slurm_scripts/job_template.job.jinja2"

parser = argparse.ArgumentParser(
    description='SLURM script generator for Horovod applications.'
)

# SLURM Variables
modules = [
        "python/3.5.2",
        "cuda10.0/blas/10.0.130",
        "cuda10.0/fft/10.0.130",
        "cuda10.0/nsight/10.0.130",
        "cuda10.0/profiler/10.0.130",
        "cuda10.0/toolkit/10.0.130",
        # For PyTorch this works
        #"cuDNN/cuda90/7.1",
        # Has to be this version for Tensorflow
        "cuDNN/cuda10.0/7.6.4",
        "openmpi/gcc/64/4.0.2",
        "nccl/cuda90/2.1.2",
    ]

# Argument validation
optional = parser._action_groups.pop() # Edited this line
required = parser.add_argument_group('required arguments')

# Required arguments
required.add_argument(
    "--jinja_template", 
    default=slurm_template,
    type=str, 
    help="""Path to Jinja template used to generate 
    SLURM Script""",
    required=True)
required.add_argument(
    "--nodes", 
    type=str, 
    help="""CSV list of GPU enabled nodes on DAS5 cluster. 
    Example: node001,node002""",
    required=True)
required.add_argument(
    "--gpus_per_node",
    type=int,
    help="Number of GPUs per node.",
    required=True
)
required.add_argument(
    "--script_path",
    type=str,
    help="""Path to Python3 script to execute with script arguments.
    Example: "~/tensorflow_synthetic_benchmark.py --batch_size=32"
    """,
    required=True
)

# Optional arguments
optional.add_argument(
    "--timeout",
    default=None,
    type=str,
    help="""After this timeframe all processes will be killed. 
    Format: D-HH:MM:SS.""",
    nargs='?'
)
optional.add_argument(
    "--disable_ib",
    default=0,
    type=int,
    help="Flag to disable InifiBand",
    nargs='?'
)
optional.add_argument(
    "--disable_p2p",
    default=0,
    type=int,
    help="Flag to disable GPU P2P Communication",
    nargs='?'
)

parser._action_groups.append(optional) # added this line

args = parser.parse_args()

# Formatted variable:
with open(args.jinja_template) as file_:
    template = Template(file_.read())

print(template.render(
    timeout=args.timeout,
    disable_ib=args.disable_ib,
    modules=modules,
    node_count=len(args.nodes.split(",")),
    nodes=args.nodes,
    gpus_per_node=args.gpus_per_node,
    script_path=args.script_path,
))
