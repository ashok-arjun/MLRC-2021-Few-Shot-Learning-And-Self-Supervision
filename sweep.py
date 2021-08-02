import wandb
import argparse
import json
import yaml

parser = argparse.ArgumentParser()

parser.add_argument("--yaml", required=True)
parser.add_argument("--count", default=10, type=int)
parser.add_argument("--project", default="FSL-SSL")
parser.add_argument("--gpu", required=True)

args = parser.parse_args()

with open(args.yaml) as file:
    sweep_config = yaml.safe_load(file)

sweep_config["parameters"]["device"] = {"values": [args.gpu]}

# print(sweep_config)

sweep_id = wandb.sweep(sweep_config, entity="meta-learners", project = args.project)
wandb.agent(sweep_id, count=args.count)