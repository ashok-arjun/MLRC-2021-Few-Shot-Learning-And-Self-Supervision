## Creates and runs a wandb sweep

import wandb

config_dict = {
    'program': 'train.py',
    'method': 'random',
    'name': 'miniImagenet Protonet sweep', # NOTE: varies
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters': {
        'lr': {'values': [1e-2, 1e-3, 1e-4, 2e-2, 2e-3, 2e-4, 3e-2, 3e-3, 3e-4]},
        'stop_epoch': {'values': [150]},
        'optimization': {'values': ['Adam', 'SGD']},
        'bn_type': {'values': [1, 2, 3]},
        'train_aug': {'values': [True, False]},
        # constants
        'dataset': {'values': ['miniImagenet']},
        'method': {'values': ['protonet']},
        'model': {'values': ['resnet18']},
        'amp': {'values': [True]}
    }
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(config_dict, entity="meta-learners", project="FSL-SSL")
    wandb.agent(sweep_id, count=25)