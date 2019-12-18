import argparse
from jinja2 import Template

parser = argparse.ArgumentParser(
    description='SLURM script generator for Horovod applications.'
)

# Script arguments
parser.add_argument(
    "--jinja_template", 
    type=str, 
    help="Path to the Jinja template.",
)
parser.add_argument(
    "-t",
    "--timeout",
    dest="timeout",
    default=None,
    type=str,
    help="""After this timeframe all processes will be killed. 
    Format: D-HH:MM:SS.""",
    nargs='?'
)
parser.add_argument(
    "--nodes", 
    type=str, 
    help="""CSV list of GPU enabled nodes on DAS5 cluster. 
    Example: node001,node002""",
)
parser.add_argument(
    "--gpus_per_node",
    type=int,
    help="Number of GPUs per node."
)
parser.add_argument(
    "--script",
    type=str,
    help="Path to Python3 script to execute."
)

args = parser.parse_args()

# Formatted variable:
print(args.jinja_template)

with open(args.jinja_template) as file_:
    template = Template(file_.read())

print(template.render(
    timeout=args.timeout,
    node_count=len(args.nodes.split(",")),
    nodes=args.nodes,
    gpus_per_node=args.gpus_per_node
))
