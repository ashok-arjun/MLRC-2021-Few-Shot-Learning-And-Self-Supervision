## Creates and runs a wandb sweep

import wandb

config_dict = {
    'program': 'train.py',
    'method': 'random',
    'name': 'miniImagenet Protonet jigsaw sweep - final', # NOTE: varies
    'metric': {'goal': 'maximize', 'name': 'val/acc'},
    'parameters': {
        'lr': {'values': [2e-3, 3e-3, 4e-3, 2e-4, 3e-4, 4e-4]},
        'stop_epoch': {'values': [100]},
        'optimization': {'values': ['Adam']},
        'bn_type': {'values': [1, 2, 3]},
        'train_aug': {'values': [True, False]},
        'jigsaw': {'values': [True]},
        'lbda': {'values': [0.5]},
        # constants
        'dataset': {'values': ['miniImagenet']},
        'method': {'values': ['protonet']},
        'model': {'values': ['resnet18']},
        'amp': {'values': [False]},
        "run_type": {'values': [0]}
    }
}

# TO ADD

# sup_ratio, semi_sup, params.dataset_combine

if __name__ == "__main__":
    sweep_id = wandb.sweep(config_dict, entity="meta-learners", project="FSL-SSL")
    wandb.agent(sweep_id, count=15)