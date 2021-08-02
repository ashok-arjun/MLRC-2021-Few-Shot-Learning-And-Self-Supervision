
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import time
import os
import glob
import random
import sys

from utils.io_utils import set_seed, parse_args

params = parse_args('test')

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler

set_seed(params.seed)

import config.configs as configs
import models.backbone as backbone

from data.datamgr_2loss import SimpleDataManager, SetDataManager
from methods.protonet_2loss import ProtoNet
    
from utils.io_utils import model_dict, get_resume_file, get_best_file, get_assigned_file
import json
from models.model_resnet import *
from utils.utils import RunningAverage, Logger, wandb_restore_models
from tqdm import tqdm

import wandb

from data.cdfsl import Chest_few_shot
from data.cdfsl import CropDisease_few_shot
from data.cdfsl import EuroSAT_few_shot
from data.cdfsl import ISIC_few_shot

import csv

torch.cuda.set_device(0)

out_file = open("cdfsl_specific_results.txt", "a")
log_file = open("cdfsl_specific_results_logs.txt", "a")

timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())                    

datamanagers = {"ISIC": ISIC_few_shot.SetDataManager, "EuroSAT": EuroSAT_few_shot.SetDataManager, \
    "ChestX": Chest_few_shot.SetDataManager, "CropDisease": CropDisease_few_shot.SetDataManager}            

# datamanagers = {"CropDisease": CropDisease_few_shot.SetDataManager} 

dataloaders = {}

for dset in datamanagers.keys():
    dataloaders[dset] = {}

    datamgr = datamanagers[dset](224, n_query = 16, n_eposide = 600, n_way = 5, n_support = 5)
    dataloaders[dset]["224"] = datamgr.get_data_loader(aug=False)


with open('runs_cdfsl.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        id = row[0]

        print(id)

        wandb.init(project="CDFSL", entity="meta-learners", id=id, resume=True) # NOTE: Change when project="CDFSL"

        dir = wandb.config["checkpoint_dir"]
        dir = dir[dir.index("results"):]

        if len(id) == 0 or len(dir) == 0:
            continue

        image_size = wandb.config["image_size"]
        model_type = wandb.config["model"]

        dataset_unlabel = wandb.config["dataset_unlabel"]

        params = wandb.config

        model = ProtoNet( model_dict[model_type], n_way=5, n_support=5, use_bn=(not params["no_bn"]), pretrain=params["pretrain"], tracking=params["tracking"],)

        try:   
            for file in ["last_model.tar"]:
                
                full_path = os.path.join(dir, file)
                pth = wandb.restore(full_path)

                print("Restored %s" % (pth.name))

                tmp = torch.load(pth.name)
                state = tmp['state']
                state_keys = list(state.keys())
                for i, key in enumerate(state_keys):
                    if "feature." in key:
                        newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                        state[newkey] = state.pop(key)
                    else:
                        state.pop(key)

                model.feature.load_state_dict(state)

                model = model.cuda()
                model.feature = model.feature.cuda()

                model.feature.eval()
                model.eval()

                dset = dataset_unlabel

                print(dataset_unlabel, end=": ")
                
                acc_mean, acc_std = model.test_loop( dataloaders[dataset_unlabel][str(image_size)], proto_only=True)  

                acc_str_c = '%4.2f%% +- %4.2f%%' %(acc_mean, 1.96* acc_std/np.sqrt(600))

                wandb.log({"test/acc_%s" % ("best" if file=="best_model.tar" else "resume") : acc_str_c})

                exp_setting = 'Time: %s, W&B ID: %s, Dataset: %s' %(timestamp, id, dset)
                acc_str = 'Test Acc: %s' %(acc_str_c)
                out_file.write( '%s %s\n' %(exp_setting,acc_str)  )                                         

                print("Removed %s" % (pth.name))
                os.remove(pth.name)

            wandb.finish()

        except ValueError as ve:
            print(ve)
            log_file.write("ValueError for %s: %s" % (id, ve))

        except RuntimeError as re:
            print(re)
            log_file.write("RuntimeError for %s: %s" % (id, re))

        except:
            print("Unexpected error:", sys.exc_info()[0])

        wandb.finish()


