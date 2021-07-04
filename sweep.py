## Creates and runs a wandb sweep

import wandb

config_dict = {
    'name': 'Sweep_PN+MIN'
    'program': 'train.py',
    'method': 'random',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters': {
        'lr': {'values': [1e-2, 1e-3, 1e-4, 2e-2, 2e-3, 2e-4, 3e-2, 3e-3, 3e-4]},
        'stop_epoch': {'values': [150]},
        'optimization': {'values': ['Adam', 'SGD']},
        'bn_type': {'values': [1, 2, 3]}       
    }
    'run_name': 'PN+MIN_Sweep_Run'
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(config_dict, entity="meta-learners", project="FSL-SSL")
    wandb.agent(sweep_id, count=25)