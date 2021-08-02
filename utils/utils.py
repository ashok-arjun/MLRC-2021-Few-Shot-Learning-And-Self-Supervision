import torch
import numpy as np
import os

import wandb
from dagshub import DAGsHubLogger


def one_hot(y, num_class):         
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

def DBindex(cl_data_file):
    class_list = cl_data_file.keys()
    cl_num= len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append( np.mean(cl_data_file[cl], axis = 0) )
        stds.append( np.sqrt(np.mean( np.sum(np.square( cl_data_file[cl] - cl_means[-1]), axis = 1))))

    mu_i = np.tile( np.expand_dims( np.array(cl_means), axis = 0), (len(class_list),1,1) )
    mu_j = np.transpose(mu_i,(1,0,2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis = 2))
    
    for i in range(cl_num):
        DBs.append( np.max([ (stds[i]+ stds[j])/mdists[i,j]  for j in range(cl_num) if j != i ]) )
    return np.mean(DBs)

def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x!=0) for x in cl_data_file[cl] ])  ) 

    return np.mean(cl_sparsity) 

class RunningAverage():
    def __init__(self):
        self.count = 0
        self.sum = 0

    def update(self, value, n_items = 1):
        self.sum += value * n_items
        self.count += n_items

    def __call__(self):
        return self.sum/self.count   

class Logger():

    wandb = None
    DAGsHub = None
    logger = None

    @staticmethod
    def init(wandb=True, DAGsHub=True, ckpt_dir=None):
        Logger.wandb = wandb
        Logger.DAGsHub = DAGsHub

        if DAGsHub:
            Logger.logger = DAGsHubLogger(metrics_path=os.path.join(ckpt_dir, "metrics.csv"), hparams_path=os.path.join(ckpt_dir,"params.yml"))
    
    @staticmethod
    def log(dict, step=None):
        if Logger.wandb:
            wandb.log(dict, step=step) if step else wandb.log(dict)
        if Logger.DAGsHub:
            Logger.logger.log_metrics(dict, step_num=step) if step else Logger.logger.log_metrics(dict)

    @staticmethod
    def log_hyperparams(params):
        if Logger.DAGsHub:
            Logger.logger.log_hyperparams(params)

def wandb_restore_models(ckpt_dir):
    last_model_path = os.path.join(ckpt_dir, "last_model.tar")
    fn = wandb.restore(last_model_path)

    # best_model_path = os.path.join(ckpt_dir, "best_model.tar")
    # wandb.restore(best_model_path)

    print("Restored models")

    return fn.name