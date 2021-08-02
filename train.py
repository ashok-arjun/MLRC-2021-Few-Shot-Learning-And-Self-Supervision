
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import time
import os
import glob
import random

from utils.io_utils import set_seed, parse_args

params = parse_args('train')
os.environ["CUDA_VISIBLE_DEVICES"] = params.device

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
from methods.maml import MAML

    
from utils.io_utils import model_dict, get_resume_file, get_best_file, get_assigned_file
import json
from models.model_resnet import *
from utils.utils import RunningAverage, Logger, wandb_restore_models
from tqdm import tqdm

import wandb
from dagshub import DAGsHubLogger

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    print("AMP is not installed. If --amp is True, code will fail.")    

# CDFSL

from data.cdfsl import Chest_few_shot
from data.cdfsl import CropDisease_few_shot
from data.cdfsl import EuroSAT_few_shot
from data.cdfsl import ISIC_few_shot

# End CDFSL

def train(base_loader, val_loader, model, optimizer, start_epoch, stop_epoch, params, base_loader_u, val_loader_u, semi_sup):    
    
    if params.amp:
        print("-----------Using mixed precision-----------") 
        model, optimizer = amp.initialize(model, optimizer)

    if params.scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) # Added by Arjun
        
    eval_interval = params.eval_interval
    max_acc = 0       
        
    pbar = tqdm(range(0, stop_epoch*len(base_loader)), total = stop_epoch*len(base_loader))
    
    pbar.update(start_epoch*len(base_loader))
    pbar.refresh()
    
    model.global_count = start_epoch*len(base_loader)
    
    for epoch in range(start_epoch,stop_epoch):
        start_time = time.time()
        
        model.train()
        avg_loss = model.train_loop(epoch, base_loader, optimizer, pbar=pbar, enable_amp=params.amp, base_loader_u=base_loader_u, semi_sup=semi_sup) 
        end_time = time.time()
        
        Logger.log({"Epoch": epoch}, step=model.global_count)
        Logger.log({"Epoch Time": end_time-start_time}, step=model.global_count)
                
        pbar.write(u'\u2713' + ' Epoch: %d; Time taken: %d sec.' % (epoch, end_time-start_time))

        if(avg_loss == float('inf') or avg_loss == 0):
            raise Exception("avg_loss is: ", avg_loss)

        if params.scheduler:
            scheduler.step()
            
        if epoch % eval_interval == 0 or epoch == stop_epoch - 1: 
            pbar.write("Validating...")
            model.eval()
            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)

            if params.jigsaw and params.rotation:
                acc, acc_jigsaw, acc_rotation = model.test_loop( val_loader, base_loader_u=val_loader_u, semi_sup=semi_sup)
                Logger.log({'val/acc': acc}, step=model.global_count)
                Logger.log({'val/acc_jigsaw': acc_jigsaw}, step=model.global_count)
                Logger.log({'val/acc_rotation': acc_rotation}, step=model.global_count)

            elif params.jigsaw:
                acc, acc_jigsaw = model.test_loop( val_loader, base_loader_u=val_loader_u, semi_sup=semi_sup)

                Logger.log({'val/acc': acc}, step=model.global_count)
                Logger.log({'val/acc_jigsaw': acc_jigsaw}, step=model.global_count)
                
            elif params.rotation:
                acc, acc_rotation = model.test_loop( val_loader, base_loader_u=val_loader_u, semi_sup=semi_sup)
                Logger.log({'val/acc': acc}, step=model.global_count)
                Logger.log({'val/acc_rotation': acc_rotation}, step=model.global_count)
                
            else:    
                acc = model.test_loop( val_loader, base_loader_u=val_loader_u, semi_sup=semi_sup)
                Logger.log({'val/acc': acc}, step=model.global_count)
                
            if acc > max_acc and params.save_model: 
                max_acc = acc
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                torch.save({'epoch':epoch, 'state':model.state_dict(), 'optimizer': optimizer.state_dict()}, outfile)
                wandb.save(outfile)

        if (((epoch) % params.save_freq==0) or (epoch==stop_epoch-1)) and params.save_model:
            outfile = os.path.join(params.checkpoint_dir, 'last_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'optimizer': optimizer.state_dict()}, outfile)
            wandb.save(outfile)

    pbar.close()
            
if __name__=='__main__':    
    # torch.cuda.set_device(int(params.device[0]))
    
    isAircraft = False

    if params.sweep:
        params.save_model = 0

    if params.bn_type == 1:
        params.no_bn = False
        params.tracking = True
    elif params.bn_type == 2:
        params.no_bn = False
        params.tracking = False
    elif params.bn_type == 3:
        params.no_bn = True
        params.tracking = False
    else:
        raise Exception("Unrecognized BN Type: ", print(params.bn_type), " of type ", type(params.bn_type))

    suffix = params.dataset_percentage 
    base_name = 'base.json' if len(suffix) == 0 else 'base_%s.json' % (suffix)

    base_file = configs.data_dir[params.dataset] + base_name
    val_file   = configs.data_dir[params.dataset] + 'val.json'
    test_file   = configs.data_dir[params.dataset] + 'novel.json'

    print("Base file: ", base_file)

    if params.demo:
        train_iter_num = 5 # NOTE: should be `100`
        val_iter_num = 5 # NOTE: should be `100`
        test_iter_num = 5 # NOTE: should be `600`
        params.stop_epoch = 1
    else:
        train_iter_num = 100 # NOTE: should be `100`
        val_iter_num = 100 # NOTE: should be `100`
        test_iter_num = 600 # NOTE: should be `600`

    image_size = params.image_size


    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    if params.method in ['protonet','matchingnet','relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
        n_query = max(1, int(params.n_query * params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        print('n_query:',n_query)
        print("semi-sup is: ", params.semi_sup)

        base_datamgr_u    = SimpleDataManager(image_size, batch_size = params.train_n_way * (params.n_shot + n_query), jigsaw=params.jigsaw, rotation=params.rotation, isAircraft=isAircraft, grey=params.grey, shuffle=True, drop_last=True)
        val_datamgr_u    = SimpleDataManager(image_size, batch_size = params.test_n_way * (params.n_shot + n_query), jigsaw=params.jigsaw, rotation=params.rotation, isAircraft=isAircraft, grey=params.grey, shuffle=True, drop_last=True)

        if params.dataset_unlabel is not None and (params.jigsaw or params.rotation):

            if "," in params.dataset_unlabel:
                params.dataset_unlabel = params.dataset_unlabel.split(",")

            suffix = params.dataset_unlabel_percentage 
            base_name = 'base.json' if len(suffix) == 0 else 'base_%s.json' % (suffix)
            val_name = 'val.json'

            if type(params.dataset_unlabel) is list:
                # a list of datasets will be there, and we need to fuse them inside get_data_loader
                print('datasets for self-supervision are: ', params.dataset_unlabel)

                base_file_u = [os.path.join('filelists', x, base_name) for x in params.dataset_unlabel]
                print("base files for self-supervision is:", base_file_u)

                val_file_u = [os.path.join('filelists', x, val_name) for x in params.dataset_unlabel]
                print("val files for self-supervision is:", val_file_u)

            else:
                print('dataset for self-supervision is: ', params.dataset_unlabel)

                if params.dataset_unlabel in ["ISIC"]:
                    base_datamgr_u         = ISIC_few_shot.SimpleDataManager(image_size, batch_size = params.train_n_way * (params.n_shot + n_query), jigsaw=params.jigsaw, rotation=params.rotation)
                    base_loader_u     = base_datamgr_u.get_data_loader(aug = params.train_aug, drop_last=True )
                    val_datamgr_u         = ISIC_few_shot.SimpleDataManager(image_size, batch_size = params.test_n_way * (params.n_shot + n_query), jigsaw=params.jigsaw, rotation=params.rotation)
                    val_loader_u     = val_datamgr_u.get_data_loader(aug = False, drop_last=True  )
                elif params.dataset_unlabel in ["EuroSAT"]:
                    base_datamgr_u         = EuroSAT_few_shot.SimpleDataManager(image_size, batch_size = params.train_n_way * (params.n_shot + n_query), jigsaw=params.jigsaw, rotation=params.rotation)
                    base_loader_u     = base_datamgr_u.get_data_loader(aug = params.train_aug, drop_last=True )
                    val_datamgr_u         = EuroSAT_few_shot.SimpleDataManager(image_size, batch_size = params.test_n_way * (params.n_shot + n_query), jigsaw=params.jigsaw, rotation=params.rotation)
                    val_loader_u     = val_datamgr_u.get_data_loader(aug = False, drop_last=True  )
                elif params.dataset_unlabel in ["CropDisease"]:
                    base_datamgr_u         = CropDisease_few_shot.SimpleDataManager(image_size, batch_size = params.train_n_way * (params.n_shot + n_query), jigsaw=params.jigsaw, rotation=params.rotation)
                    base_loader_u     = base_datamgr_u.get_data_loader(aug = params.train_aug, drop_last=True )
                    val_datamgr_u         = CropDisease_few_shot.SimpleDataManager(image_size, batch_size = params.test_n_way * (params.n_shot + n_query), jigsaw=params.jigsaw, rotation=params.rotation)
                    val_loader_u     = val_datamgr_u.get_data_loader(aug = False, drop_last=True  )
                elif params.dataset_unlabel in ["ChestX"]:
                    base_datamgr_u         = Chest_few_shot.SimpleDataManager(image_size, batch_size = params.train_n_way * (params.n_shot + n_query), jigsaw=params.jigsaw, rotation=params.rotation)
                    base_loader_u     = base_datamgr_u.get_data_loader(aug = params.train_aug, drop_last=True )
                    val_datamgr_u         = Chest_few_shot.SimpleDataManager(image_size, batch_size = params.test_n_way * (params.n_shot + n_query), jigsaw=params.jigsaw, rotation=params.rotation)
                    val_loader_u     = val_datamgr_u.get_data_loader(aug = False, drop_last=True  )
                else: 
                    base_file_u = os.path.join('filelists', params.dataset_unlabel, base_name) 
                    val_file_u = os.path.join('filelists', params.dataset_unlabel, val_name)
                    base_loader_u     = base_datamgr_u.get_data_loader( base_file_u , aug = params.train_aug)
                    val_loader_u     = val_datamgr_u.get_data_loader( val_file_u , aug = False)

        else:
            base_loader_u     = None
            val_loader_u = None

        # train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot, \
                                        jigsaw=params.jigsaw, lbda=params.lbda,  rotation=params.rotation, lbda_jigsaw=params.lbda_jigsaw, lbda_rotation=params.lbda_rotation) 
        base_datamgr            = SetDataManager(image_size, n_query = n_query, n_eposide = train_iter_num, **train_few_shot_params, isAircraft=isAircraft, grey=params.grey, low_res=params.low_res, sup_ratio=params.sup_ratio, semi_sup=params.semi_sup)
        base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
         
        val_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot, \
                                        jigsaw=params.jigsaw, lbda=params.lbda, rotation=params.rotation, lbda_jigsaw=params.lbda_jigsaw, lbda_rotation=params.lbda_rotation) 
        val_datamgr             = SetDataManager(image_size, n_query = n_query, n_eposide = val_iter_num, **val_few_shot_params, isAircraft=isAircraft, grey=params.grey, low_res=params.low_res, semi_sup=params.semi_sup)
        val_loader              = val_datamgr.get_data_loader( val_file, aug = False) 
        #a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor     


        if params.dataset_unlabel in ["ISIC"]:
            test_datamgr             = ISIC_few_shot.SetDataManager(image_size, n_query = n_query, n_eposide = test_iter_num, n_way = params.test_n_way, n_support = params.n_shot)
            test_loader              = test_datamgr.get_data_loader( aug = False)  
        elif params.dataset_unlabel in ["EuroSAT"]:
            test_datamgr             = EuroSAT_few_shot.SetDataManager(image_size, n_query = n_query, n_eposide = test_iter_num, n_way = params.test_n_way, n_support = params.n_shot)
            test_loader              = test_datamgr.get_data_loader( aug = False)  
        elif params.dataset_unlabel in ["CropDisease"]:
            test_datamgr             = CropDisease_few_shot.SetDataManager(image_size, n_query = n_query, n_eposide = test_iter_num, n_way = params.test_n_way, n_support = params.n_shot)
            test_loader              = test_datamgr.get_data_loader( aug = False)  
        elif params.dataset_unlabel in ["ChestX"]:
            test_datamgr             = Chest_few_shot.SetDataManager(image_size, n_query = n_query, n_eposide = test_iter_num, n_way = params.test_n_way, n_support = params.n_shot)
            test_loader              = test_datamgr.get_data_loader( aug = False)  
        else:
            test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot, \
                                            jigsaw=params.jigsaw, lbda=params.lbda, rotation=params.rotation, lbda_jigsaw=params.lbda_jigsaw, lbda_rotation=params.lbda_rotation) 
            test_datamgr             = SetDataManager(image_size, n_query = n_query, n_eposide = test_iter_num, **test_few_shot_params, isAircraft=isAircraft, grey=params.grey, low_res=params.low_res, semi_sup=params.semi_sup)
            test_loader              = test_datamgr.get_data_loader( test_file, aug = False)    

        if params.method == 'protonet':
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params , use_bn=(not params.no_bn), pretrain=params.pretrain, tracking=params.tracking, model=params.model)
        elif params.method in ['maml' , 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True

            BasicBlock.maml = True
            Bottleneck.maml = True
            ResNet.maml = True

            model           = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , tracking=params.tracking, **train_few_shot_params , use_bn=(not params.no_bn), pretrain=params.pretrain, model=params.model)
            if params.dataset in ['omniglot', 'cross_char']: #maml use different parameter in omniglot
                model.n_task     = 32
                model.task_update_num = 1
                model.train_lr = 0.1
    else:
       raise ValueError('Unknown method')
    # import ipdb; ipdb.set_trace()
    # import ipdb; ipdb.set_trace()

    if params.json_seed is not None:
        params.checkpoint_dir = '%s/results/%s_%s/%s_%s_%s' %(configs.save_dir, params.dataset, params.json_seed, params.date, params.model, params.method)
    else:
        params.checkpoint_dir = '%s/results/%s/%s_%s_%s' %(configs.save_dir, params.dataset, params.date, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    params.checkpoint_dir += '_%dway_%dshot_%dquery' %( params.train_n_way, params.n_shot, params.n_query)

    params.checkpoint_dir += '_%d'%image_size
    
    ## Track bn stats
    if params.tracking:
        params.checkpoint_dir += '_tracking'

    ## Use subset of training data
    if params.firstk > 0:
        params.checkpoint_dir += ('_first'+str(params.firstk))

    ## Use grey image
    if params.grey:
        params.checkpoint_dir += '_grey'

    ## Use low_res image
    if params.low_res:
        params.checkpoint_dir += '_low_res'

    ## Add jigsaw and rotation
    if params.jigsaw and params.rotation:
        params.checkpoint_dir += '_jigsaw_lbda%.2f_rotation_lbda%.2f'%(params.lbda_jigsaw, params.lbda_rotation)
    ## Add jigsaw
    elif params.jigsaw:
        params.checkpoint_dir += '_jigsaw_lbda%.2f'%(params.lbda)
    ## Add rotation
    elif params.rotation:
        params.checkpoint_dir += '_rotation_lbda%.2f'%(params.lbda)

    if params.semi_sup:
        params.checkpoint_dir += '_semi_sup%.2f'%(params.lbda)

    if params.dataset_unlabel:
        params.checkpoint_dir += '_dataset_unlabel=%s'%("".join(params.dataset_unlabel))

    params.checkpoint_dir += '_sup_ratio=%d'%(params.sup_ratio)

    params.checkpoint_dir += params.optimization

    params.checkpoint_dir += '_lr%.4f'%(params.lr)
    if params.finetune:
        params.checkpoint_dir += '_finetune'

    if params.random:
        params.checkpoint_dir += 'results/'+params.dataset+'/random'
    if params.debug:
        params.checkpoint_dir += 'results/'+params.dataset+'/debug'

    params.checkpoint_dir += 'dset_percent='+params.dataset_percentage
    params.checkpoint_dir += 'dset_unlabel_percent='+params.dataset_unlabel_percentage


    print('Checkpoint path:',params.checkpoint_dir)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx' :
        stop_epoch = params.stop_epoch * model.n_task #maml use multiple tasks in one update 

    if params.resume:

        wandb.init(project=params.project, entity="meta-learners", id=params.run_id, resume=True)
        wandb.watch(model)
        Logger.init(wandb=True, DAGsHub=True, ckpt_dir=params.checkpoint_dir)

        old_ckpt_dir = wandb.config["checkpoint_dir"]
        old_ckpt_dir = old_ckpt_dir[old_ckpt_dir.index("results"):]

        params.checkpoint_dir = old_ckpt_dir

        resume_file = wandb_restore_models(old_ckpt_dir)
        # resume_file = get_resume_file(old_ckpt_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file, map_location="cpu")
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])

        print("Resuming run %s from epoch %d" % (params.run_name, start_epoch))

    elif params.loadfile != '':
        print('Loading model from: ' + params.loadfile)
        checkpoint = torch.load(params.loadfile, map_location='cpu')
        model.load_state_dict(checkpoint['state'])

    model = model.cuda()
    model.feature = model.feature.cuda()

    if params.optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    elif params.optimization == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr)
    elif params.optimization == 'Nesterov':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, nesterov=True, momentum=0.9, weight_decay=params.wd)
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    if params.resume:
        optimizer.load_state_dict(tmp['optimizer'])

    if not params.only_test:

        if not params.resume:

            json.dump(vars(params), open(params.checkpoint_dir+'/configs.json','w'))        
            wandb.init(config=vars(params), project=params.project, entity="meta-learners")        
            wandb.run.name = wandb.run.id if not params.run_name else params.run_name        
            wandb.watch(model)        

            Logger.init(wandb=True, DAGsHub=True, ckpt_dir=params.checkpoint_dir)
            Logger.log_hyperparams(vars(params))

        train(base_loader, val_loader,  model, optimizer, start_epoch, stop_epoch, params, base_loader_u, val_loader_u, params.semi_sup)
    
    if not params.only_train:

        test_accs = []

        for i,fn in enumerate([get_resume_file, get_best_file]):

            print(fn)

            split = 'novel'
            if params.save_iter != -1:
                split_str = split + "_" +str(params.save_iter)
            else:
                split_str = split

            few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)
            acc_all = []

            if params.loadfile != '':
                modelfile   = params.loadfile
                checkpoint_dir = params.loadfile
            else:
                checkpoint_dir = params.checkpoint_dir
                if params.save_iter != -1:
                    modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
                else:
                    modelfile   = fn(checkpoint_dir)

            if params.method in ['maml', 'maml_approx']:
                if modelfile is not None:
                    tmp = torch.load(modelfile)
                    state = tmp['state']
                    state_keys = list(state.keys())
                    for i, key in enumerate(state_keys):
                        if "feature." in key:
                            newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                            state[newkey] = state.pop(key)
                        else:
                            state.pop(key)
                    # model.load_state_dict(tmp['state'], strict=False)
                    model.feature.load_state_dict(tmp['state'])

                datamgr         = SetDataManager(image_size, n_eposide = test_iter_num, n_query = params.n_query , **few_shot_params, isAircraft=isAircraft, grey=params.grey, low_res=params.low_res)
                loadfile = configs.data_dir[params.dataset] + split + '.json'
                novel_loader     = datamgr.get_data_loader( loadfile, aug = False)
                if params.adaptation:
                    model.task_update_num = 100 #We perform adaptation on MAML simply by updating more times.
                model.eval()

                acc_mean, acc_std = model.test_loop( test_loader, semi_sup=params.semi_sup, proto_only=True, std_also=True)        

                Logger.log({"test/acc_%s" % ("resume" if fn==get_resume_file else "best"): acc_mean})
                test_accs.append(acc_mean)

                out_dir = os.path.join( checkpoint_dir)

                os.makedirs(out_dir, exist_ok=True)

                with open(os.path.join( checkpoint_dir, split_str +"_test.txt") , 'a') as f:
                    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
                    aug_str = '-aug' if params.train_aug else ''
                    aug_str += '-adapted' if params.adaptation else ''
                    exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str , params.n_shot , params.train_n_way, params.test_n_way )
                    acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(test_iter_num, acc_mean, 1.96* acc_std/np.sqrt(test_iter_num))
                    f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_str)  )
            else:

                tmp = torch.load(modelfile)
                state = tmp['state']
                state_keys = list(state.keys())
                for i, key in enumerate(state_keys):
                    if "feature." in key:
                        newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                        state[newkey] = state.pop(key)
                    else:
                        state.pop(key)

                model.feature.load_state_dict(state)
                model.feature.eval()
                model = model.cuda()
                model.feature = model.feature.cuda()
                model.eval()

                if params.semi_sup:
                    print("Performing supervised + semi-supervised inference...")
                else:
                    print("Performing inference...")

                acc_mean, acc_std = model.test_loop( test_loader, semi_sup=params.semi_sup, proto_only=True)        

                Logger.log({"test/acc_%s" % ("resume" if fn==get_resume_file else "best"): acc_mean})
                test_accs.append(acc_mean)

                out_dir = os.path.join( checkpoint_dir)

                os.makedirs(out_dir, exist_ok=True)

                with open(os.path.join( checkpoint_dir, split_str +"_test.txt") , 'a') as f:
                    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
                    aug_str = '-aug' if params.train_aug else ''
                    aug_str += '-adapted' if params.adaptation else ''
                    exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str , params.n_shot , params.train_n_way, params.test_n_way )
                    acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(test_iter_num, acc_mean, 1.96* acc_std/np.sqrt(test_iter_num))
                    f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_str)  )
        
        Logger.log({"test/acc": test_accs[1]})
        wandb.save(os.path.join(params.checkpoint_dir, "last_model.tar"))
        wandb.save(os.path.join(params.checkpoint_dir, "best_model.tar"))

        wandb.finish()

    if params.save_model:
        os.remove(os.path.join(params.checkpoint_dir, "last_model.tar"))
        os.remove(os.path.join(params.checkpoint_dir, "best_model.tar"))
