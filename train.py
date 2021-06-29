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
    
from utils.io_utils import model_dict, get_resume_file, get_best_file, get_assigned_file
import json
from models.model_resnet import *
from utils.utils import RunningAverage
from tqdm import tqdm

import wandb

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    print("AMP is not installed. If --amp is True, code will fail.")    

def train(base_loader, val_loader, model, optimizer, start_epoch, stop_epoch, params, base_loader_u = None, semi_sup=False, self_sup_origin="own"):    
    
    if params.amp:
        print("-----------Using mixed precision-----------") 
        model, optimizer = amp.initialize(model, optimizer)

    eval_interval = params.eval_interval
    max_acc = 0       
        
    pbar = tqdm(range(0, stop_epoch*len(base_loader)), total = stop_epoch*len(base_loader))
    
    pbar.update(start_epoch*len(base_loader))
    pbar.refresh()
    
    model.global_count = start_epoch*len(base_loader)
    
    for epoch in range(start_epoch,stop_epoch):
        start_time = time.time()
        
        model.train()
        avg_loss = model.train_loop(epoch, base_loader, optimizer, pbar=pbar, enable_amp=params.amp, base_loader_u=base_loader_u, semi_sup=semi_sup, self_sup_origin=self_sup_origin) 
        
        end_time = time.time()
        
        wandb.log({"Epoch": epoch}, step=model.global_count)
        wandb.log({"Epoch Time": end_time-start_time}, step=model.global_count)        
        
        pbar.write(u'\u2713' + ' Epoch: %d; Time taken: %d sec.' % (epoch, end_time-start_time))

        if(avg_loss == float('inf') or avg_loss == 0):
            raise Exception("avg_loss is: ", avg_loss)

        if epoch % eval_interval == 0 or epoch == stop_epoch - 1: 
            model.eval()
            if not os.path.isdir(params.checkpoint_dir):
                os.makedirs(params.checkpoint_dir)

            if params.jigsaw and params.rotation:
                acc, acc_jigsaw, acc_rotation = model.test_loop( val_loader, base_loader_u=base_loader_u, semi_sup=semi_sup, self_sup_origin=self_sup_origin)
                wandb.log({'val/acc': acc}, step=model.global_count)
                wandb.log({'val/acc_jigsaw': acc_jigsaw}, step=model.global_count)
                wandb.log({'val/acc_rotation': acc_rotation}, step=model.global_count)

            elif params.jigsaw:
                acc, acc_jigsaw = model.test_loop( val_loader, base_loader_u=base_loader_u, semi_sup=semi_sup, self_sup_origin=self_sup_origin)

                wandb.log({'val/acc': acc}, step=model.global_count)
                wandb.log({'val/acc_jigsaw': acc_jigsaw}, step=model.global_count)

            elif params.rotation:
                acc, acc_rotation = model.test_loop( val_loader, base_loader_u=base_loader_u, semi_sup=semi_sup, self_sup_origin=self_sup_origin)
                wandb.log({'val/acc': acc}, step=model.global_count)
                wandb.log({'val/acc_rotation': acc_rotation}, step=model.global_count)
            else:    
                acc = model.test_loop( val_loader, base_loader_u=base_loader_u, semi_sup=semi_sup, self_sup_origin=self_sup_origin)
                wandb.log({'val/acc': acc}, step=model.global_count)
            if acc > max_acc : 
                max_acc = acc
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                torch.save({'epoch':epoch, 'state':model.state_dict(), 'optimizer': optimizer.state_dict()}, outfile)
                wandb.save(outfile)

        if ((epoch) % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, 'last_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict(), 'optimizer': optimizer.state_dict()}, outfile)
            wandb.save(outfile)

    pbar.close()
            
if __name__=='__main__':    
    torch.cuda.set_device(int(params.device[0])) 

    isAircraft = (params.dataset == 'aircrafts')

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

    if params.dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json' 
        val_file   = configs.data_dir['CUB'] + 'val.json' 
    elif params.dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json' 
        val_file   = configs.data_dir['emnist'] + 'val.json' 
    elif params.dataset == 'flowers_original':
        base_file = configs.data_dir['flowers'] + 'base.json'
        val_file = configs.data_dir['flowers'] + 'val.json'
    elif params.dataset == 'dogs_original':
        base_file = configs.data_dir['dogs'] + 'base.json'
        val_file = configs.data_dir['dogs'] + 'val.json'
    elif params.dataset == 'aircrafts_original':
        base_file = configs.data_dir['aircrafts'] + 'base.json'
        val_file = configs.data_dir['aircrafts'] + 'val.json'
    elif params.dataset == 'cars_original':
        base_file = configs.data_dir['cars'] + 'base.json'
        val_file = configs.data_dir['cars'] + 'val.json'
    elif params.dataset == 'CUB_original':
        if params.firstk > 0:
            base_file = configs.data_dir['CUB'] + 'original_split_train_first_'+str(params.firstk)+'.json'
        else:
            base_file = configs.data_dir['CUB'] + 'base.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif params.dataset == 'CUB_subset':
        base_file = configs.data_dir['CUB'] + 'original_split_train_base.json'
        val_file = configs.data_dir['CUB'] + 'original_split_train_val.json'
    elif '_' in params.dataset:
        base_file = configs.data_dir[params.dataset.split('_')[0]] + 'base_' + params.dataset.split('_')[1] +'.json'
        print("base file for labeled dataset is:", base_file)
        val_file = configs.data_dir[params.dataset.split('_')[0]] + 'val.json'
    else:
        if params.json_seed is not None:
            base_file = configs.data_dir[params.dataset] + 'base' + params.json_seed + '.json' 
            val_file   = configs.data_dir[params.dataset] + 'val' + params.json_seed + '.json' 
        else:
            base_file = configs.data_dir[params.dataset] + 'base.json' 
            val_file   = configs.data_dir[params.dataset] + 'val.json' 

    if 'Conv' in params.model:
        if params.dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            # image_size = 84
            image_size = 255
    else:
        # image_size = 224 #original setting
        # image_size = 256 #my setting
        image_size = params.image_size


    if params.dataset in ['omniglot', 'cross_char']:
        assert params.model == 'Conv4' and not params.train_aug ,'omniglot only support Conv4 without augmentation'
        params.model = 'Conv4S'

    if params.method in ['baseline', 'baseline++'] :
        # base_datamgr    = SimpleDataManager(image_size, batch_size = 16, jigsaw=params.jigsaw, rotation=params.rotation, isAircraft=isAircraft, grey=params.grey)
        base_datamgr    = SimpleDataManager(image_size, batch_size = params.bs, jigsaw=params.jigsaw, rotation=params.rotation, isAircraft=isAircraft, grey=params.grey)
        base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        # val_datamgr     = SimpleDataManager(image_size, batch_size = 64, jigsaw=params.jigsaw, rotation=params.rotation, isAircraft=isAircraft, grey=params.grey)
        val_datamgr     = SimpleDataManager(image_size, batch_size = params.bs, jigsaw=params.jigsaw, rotation=params.rotation, isAircraft=isAircraft, grey=params.grey)
        val_loader      = val_datamgr.get_data_loader( val_file, aug = False)
        
        # if params.dataset == 'omniglot':
        #     assert params.num_classes >= 4112, 'class number need to be larger than max label id in base class'
        # if params.dataset == 'cross_char':
        #     assert params.num_classes >= 1597, 'class number need to be larger than max label id in base class'
        if params.dataset == 'CUB_original':
            params.num_classes = 200
        elif params.dataset == 'cars_original':
            params.num_classes = 196
        elif params.dataset == 'aircrafts_original':
            params.num_classes = 100
        elif params.dataset == 'dogs_original':
            params.num_classes = 120
        elif params.dataset == 'flowers_original':
            params.num_classes = 102
        elif params.dataset == 'miniImagenet':
            params.num_classes = 100
        elif params.dataset == 'tieredImagenet':
            params.num_classes = 608


        if params.method == 'baseline':
            model           = BaselineTrain( model_dict[params.model], params.num_classes, \
                                            jigsaw=params.jigsaw, lbda=params.lbda, rotation=params.rotation, tracking=params.tracking)
        elif params.method == 'baseline++':
            model           = BaselineTrain( model_dict[params.model], params.num_classes, \
                                            loss_type = 'dist', jigsaw=params.jigsaw, lbda=params.lbda, rotation=params.rotation, tracking=params.tracking)

    elif params.method in ['protonet','matchingnet','relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
        n_query = max(1, int(params.n_query * params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        print('n_query:',n_query)

        base_datamgr_u    = SimpleDataManager(image_size, batch_size = params.bs, jigsaw=params.jigsaw, rotation=params.rotation, isAircraft=isAircraft, grey=params.grey, shuffle=True)
        if params.dataset_unlabel is not None:
            base_file_unlabel = os.path.join('filelists', params.dataset_unlabel, 'base.json')
            print("base file for unlabeled dataset is:", base_file_unlabel)
            base_loader_u     = base_datamgr_u.get_data_loader( base_file_unlabel , aug = params.train_aug )
        else:
            base_loader_u     = None
 
        # train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot, \
                                        jigsaw=params.jigsaw, lbda=params.lbda,  rotation=params.rotation, lbda_jigsaw=params.lbda_jigsaw, lbda_rotation=params.lbda_rotation) 
        base_datamgr            = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params, isAircraft=isAircraft, grey=params.grey, low_res=params.low_res)
        base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
         
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot, \
                                        jigsaw=params.jigsaw, lbda=params.lbda, rotation=params.rotation, lbda_jigsaw=params.lbda_jigsaw, lbda_rotation=params.lbda_rotation) 
        val_datamgr             = SetDataManager(image_size, n_query = n_query, n_eposide = 600, **test_few_shot_params, isAircraft=isAircraft, grey=params.grey, low_res=params.low_res)
        val_loader              = val_datamgr.get_data_loader( val_file, aug = False) 
        #a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor        

        if params.method == 'protonet':
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params , use_bn=(not params.no_bn), pretrain=params.pretrain, tracking=params.tracking)
        elif params.method in ['maml' , 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True

            BasicBlock.maml = True
            Bottleneck.maml = True
            ResNet.maml = True

            model           = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , tracking=params.tracking, **train_few_shot_params )
            if params.dataset in ['omniglot', 'cross_char']: #maml use different parameter in omniglot
                model.n_task     = 32
                model.task_update_num = 1
                model.train_lr = 0.1
    else:
       raise ValueError('Unknown method')
    # import ipdb; ipdb.set_trace()
    model = model.cuda()
    model.feature = model.feature.cuda()
    # import ipdb; ipdb.set_trace()

    if params.optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    elif params.optimization == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr)
    elif params.optimization == 'Nesterov':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, nesterov=True, momentum=0.9, weight_decay=params.wd)
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    if params.json_seed is not None:
        params.checkpoint_dir = '%s/checkpoints/%s_%s/%s_%s_%s' %(configs.save_dir, params.dataset, params.json_seed, params.date, params.model, params.method)
    else:
        params.checkpoint_dir = '%s/checkpoints/%s/%s_%s_%s' %(configs.save_dir, params.dataset, params.date, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot_%dquery' %( params.train_n_way, params.n_shot, params.n_query)

    params.checkpoint_dir += '_%d'%image_size
    
    if params.dataset_unlabel is not None:
        params.checkpoint_dir += params.dataset_unlabel
        # params.checkpoint_dir += str(params.bs)

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

    params.checkpoint_dir += params.optimization

    params.checkpoint_dir += '_lr%.4f'%(params.lr)
    if params.finetune:
        params.checkpoint_dir += '_finetune'

    if params.random:
        params.checkpoint_dir = 'checkpoints/'+params.dataset+'/random'
    if params.debug:
        params.checkpoint_dir = 'checkpoints/'+params.dataset+'/debug'

    print('Checkpoint path:',params.checkpoint_dir)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx' :
        stop_epoch = params.stop_epoch * model.n_task #maml use multiple tasks in one update 
    ## Use Google paper setting
    if params.method == 'baseline' and 'original' in params.dataset:
        stop_epoch = int(20000/len(base_loader))
        print('train 20000 iters which is '+str(stop_epoch)+' epoch')

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])
            optimizer.load_state_dict(tmp['optimizer'])
            del tmp

        if not params.run_name:
            raise Exception("Resume run name not given.")

        print("Resuming run %s from epoch %d" % (params.run_name, start_epoch))

        wandb.init(config=vars(params), project="FSL-SSL", entity="meta-learners", id=params.run_name, resume=True)        
        wandb.watch(model)

    elif params.warmup: #We also support warmup from pretrained baseline feature, but we never used in our paper
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None: 
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')
    
    if params.loadfile != '':
        print('Loading model from: ' + params.loadfile)
        checkpoint = torch.load(params.loadfile)
        ## remove last layer for baseline
        pretrained_dict = {k: v for k, v in checkpoint['state'].items() if 'classifier' not in k and 'loss_fn' not in k}
        # import ipdb; ipdb.set_trace()
        # print(pretrained_dict)
        print('Load model from:',params.loadfile)
        model.load_state_dict(pretrained_dict, strict=False)

    if not params.resume:

        json.dump(vars(params), open(params.checkpoint_dir+'/configs.json','w'))        
        wandb.init(config=vars(params), project="FSL-SSL", entity="meta-learners")        
        wandb.run.name = wandb.run.id if not params.run_name else params.run_name        
        wandb.watch(model)    
    
    train(base_loader, val_loader,  model, optimizer, start_epoch, stop_epoch, params, base_loader_u=base_loader_u, semi_sup=params.semi_sup, self_sup_origin="unlabel" if params.dataset_unlabel else "none")


    ##### save_features (except maml) and test, added by me #####
    split = 'novel'
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split

    iter_num = 600
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot)
    acc_all = []

    if params.loadfile != '':
        modelfile   = params.loadfile
        checkpoint_dir = params.loadfile
    else:
        checkpoint_dir = params.checkpoint_dir
        if params.save_iter != -1:
            modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
        elif params.method in ['baseline', 'baseline++'] :
            modelfile   = get_resume_file(checkpoint_dir)
        else:
            modelfile   = get_best_file(checkpoint_dir)

    # if not params.method in ['baseline', 'baseline++'] :
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
        print('modelfile:',modelfile)

        datamgr         = SetDataManager(image_size, n_eposide = iter_num, n_query = 15 , **few_shot_params, isAircraft=isAircraft, grey=params.grey, low_res=params.low_res)
        loadfile = configs.data_dir[params.dataset] + split + '.json'
        novel_loader     = datamgr.get_data_loader( loadfile, aug = False)
        if params.adaptation:
            model.task_update_num = 100 #We perform adaptation on MAML simply by updating more times.
        model.eval()
        acc_mean, acc_std = model.test_loop( novel_loader, return_std = True)
        print(acc_mean, acc_std)
    else:
        
        if params.method == 'baseline':
            model           = BaselineFinetune( model_dict[params.model], **few_shot_params )
        elif params.method == 'baseline++':
            model           = BaselineFinetune( model_dict[params.model], loss_type = 'dist', **few_shot_params )

        if params.save_iter != -1:
            outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + "_" + str(params.save_iter)+ ".hdf5")
        else:
            outfile = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + ".hdf5")

        datamgr         = SimpleDataManager(image_size, batch_size = params.test_bs, isAircraft=isAircraft, grey=params.grey, low_res=params.low_res)
        if '_' in params.dataset:
            loadfile = configs.data_dir[params.dataset.split('_')[0]] + split + '.json'
        else:
            loadfile = configs.data_dir[params.dataset] + split + '.json'
        data_loader      = datamgr.get_data_loader(loadfile, aug = False)

        tmp = torch.load(modelfile)
        state = tmp['state']
        state_keys = list(state.keys())
        for i, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                state[newkey] = state.pop(key)
            else:
                state.pop(key)


        # import ipdb; ipdb.set_trace()
        # if params.method != 'baseline':
        model.feature.load_state_dict(state)
        model.feature.eval()
        model = model.cuda()
        model.feature = model.feature.cuda()
        # else:
        #     model.load_state_dict(state)
        model.eval()

        dirname = os.path.dirname(outfile)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        # import ipdb; ipdb.set_trace()
        # outfile += '_finetune'
        print('save outfile at:', outfile)
        from save_features import save_features
        save_features(model, data_loader, outfile)

        ### from test.py ###
        from test import feature_evaluation
        novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str +".hdf5") #defaut split = novel, but you can also test base or val classes
        print('load novel file from:',novel_file)
        import data.feature_loader as feat_loader
        cl_data_file = feat_loader.init_loader(novel_file)
        
        for i in range(0, iter_num):
            acc = feature_evaluation(cl_data_file, model, n_query = 15, adaptation = params.adaptation, **few_shot_params)
            acc_all.append(acc)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        wandb.log({"test/acc": np.mean(acc_all), "episodes": iter_num})

        with open(os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str +"_test.txt") , 'a') as f:
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            aug_str = '-aug' if params.train_aug else ''
            aug_str += '-adapted' if params.adaptation else ''
            if params.method in ['baseline', 'baseline++'] :
                exp_setting = '%s-%s-%s-%s%s %sshot %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.test_n_way )
            else:
                exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str , params.n_shot , params.train_n_way, params.test_n_way )
            acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
            f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_str)  )


        
