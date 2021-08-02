import numpy as np
import random
import torch
import os
import glob
import argparse
import models.backbone as backbone
from models.model_resnet import *

model_dict = dict(
            Conv4 = backbone.Conv4,
            Conv4S = backbone.Conv4S,
            Conv6 = backbone.Conv6,
            ResNet10 = backbone.ResNet10,
            ResNet18 = backbone.ResNet18,
            ResNet34 = backbone.ResNet34,
            ResNet50 = backbone.ResNet50,
            ResNet101 = backbone.ResNet101,
            resnet18 = 'resnet18',
            resnet18_pytorch = 'resnet18_pytorch',
            resnet50_pytorch = 'resnet50_pytorch'
            ) 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset'     , default='CUB',        help='CUB/miniImagenet/cross/omniglot/cross_char')
    parser.add_argument('--model'       , default='Conv4',      help='model: Conv{4|6} / ResNet{10|18|34|50|101}') # 50 and 101 are not used in the paper
    parser.add_argument('--method'      , default='baseline',   help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/maml{_approx}') #relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
    parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training') #baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support') #baseline and baseline++ only use this parameter in finetuning
    parser.add_argument('--train_aug'   , type=str2bool, nargs='?', default=True, const=True,   help='perform data augmentation or not during training ') #still required for save_features.py and test.py to find the model path correctly

    parser.add_argument('--jigsaw'      , type=str2bool, nargs='?', default=False, const=True, help='multi-task training')
    parser.add_argument('--lbda'        , default=0.0, type=float,  help='lambda for the jigsaw loss, (1-lambda) for proto loss')
    # parser.add_argument('--lbda_proto'  , default=1.0, type=float,  help='lambda for the protonet loss')
    parser.add_argument('--lr'          , default=0.001, type=float,  help='learning rate')
    parser.add_argument('--optimization', default='Adam', type=str,  help='Adam or SGD')
    parser.add_argument('--loadfile'    , default='', type=str,  help='load pre-trained model')
    parser.add_argument('--finetune'    , action='store_true',  help='finetuning from jigsaw to protonet')

    parser.add_argument('--random'      , action='store_true',  help='random init net')

    parser.add_argument('--n_query'     , default=16, type=int,  help='number of query, 16 is used in the paper')
    parser.add_argument('--image_size'  , default=224, type=int,  help='224 is used in the paper')

    parser.add_argument('--debug'      , action='store_true',  help='')
    parser.add_argument('--json_seed'  , default=None, type=str,  help='seed for CUB split')
    parser.add_argument('--date'       , default='', type=str,  help='date of the exp')

    parser.add_argument('--rotation'    , type=str2bool, nargs='?', default=False, const=True, help='multi-task training')
    parser.add_argument('--grey'        , action='store_true',  help='use grey image')  # Use for CUB, dogs and flowers only
    parser.add_argument('--low_res', type=str2bool, nargs='?', default=False, const=True, help='semi_sup')  # Use for cars and aircrafts only

    parser.add_argument('--firstk'      , default=0, type=int, help='first k images per class for training CUB')

    parser.add_argument('--testiter'       , default=199, type=int,  help='date of the exp')

    parser.add_argument('--wd'          , default=0.01, type=float,  help='weight decay, 0.01 to 0.00001')
    parser.add_argument('--bs'          , default=16, type=int,  help='batch size for baseline, 256 for fgvc?')
    parser.add_argument('--iterations'          , default=20000, type=int,  help='number of iterations')

    parser.add_argument('--useVal'        , action='store_true',  help='use val set as test set')
    parser.add_argument('--scheduler'       , type=str2bool, nargs='?', default=False, const=True, help='lr scheduler')
    # parser.add_argument('--step_size'          , default=10000, type=int,  help='step for step scheduler')
    # parser.add_argument('--gamma'          , default=0.2, type=float,  help='gamma for step scheduler')

    parser.add_argument('--lbda_jigsaw'        , default=0.0, type=float,  help='lambda for the jigsaw loss, (1-lambda) for proto loss')
    parser.add_argument('--lbda_rotation'        , default=0.0, type=float,  help='lambda for the jigsaw loss, (1-lambda) for proto loss')

    parser.add_argument('--pretrain'    , type=str2bool, nargs='?', default=False, const=True,   help='use imagenet pre-train model')

    parser.add_argument('--dataset_unlabel'     , default=None,        help='CUB/miniImagenet/cross/omniglot/cross_char')
    parser.add_argument('--dataset_unlabel_percentage'     , default="",        help='20,40,60,80')

    parser.add_argument('--dataset_percentage'     , default="",        help='20,40,60,80')

    parser.add_argument('--bn_type', default=1, type=int, help="1 for BN+Tracking. 2 for BN + no tracking, 3 for no BN. BN --> BatchNorm")

    parser.add_argument('--test_bs'          , default=64, type=int,  help='batch size for testing w/o batchnorm')
    parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
    parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
    parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')

    parser.add_argument('--device', type=str, default="0", help='GPU')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', type=str2bool, nargs='?', default=False, const=True, help='amp') 

    if script == 'train':
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=100, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=400, type=int, help ='Stopping epoch') # for meta-learning methods, each epoch contains 100 episodes
        parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        parser.add_argument('--warmup'      , action='store_true', help='continue from baseline, neglected if resume is true') #never used in the paper

        parser.add_argument('--eval_interval', type=int, default=50, help='eval_interval') 
        parser.add_argument('--run_name', default=None, help="wandb run name")
        parser.add_argument('--run_id', default=None, help="wandb run ID")
        parser.add_argument('--semi_sup', type=str2bool, nargs='?', default=False, const=True, help='semi_sup') 
        parser.add_argument('--sup_ratio', type=float, default=1.0) 
        parser.add_argument('--only_test', type=str2bool, nargs='?', default=False, const=True) 
        parser.add_argument('--project', type=str, default="FSL-SSL")
        parser.add_argument('--save_model', type=str2bool, nargs='?', default=True, const=True)
        parser.add_argument('--demo', type=str2bool, nargs='?', default=False, const=True) # if True, train = 1 epoch, all episodes are 5 episodes (train,val,test)
        parser.add_argument('--only_train', type=str2bool, nargs='?', default=False, const=True) # if True, only train
        parser.add_argument('--sweep', type=str2bool, nargs='?', default=False, const=True) # if True, train = 1 epoch, all episodes are 5 episodes (train,val,test)

    return parser.parse_args()


def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    resume_file = os.path.join(checkpoint_dir, 'last_model.tar')
    return resume_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.inputs = list(next(self.loader))
        except StopIteration:
            self.inputs = None
            return
        with torch.cuda.stream(self.stream):
            for i,tensor in enumerate(self.inputs):
                self.inputs[i] = self.inputs[i].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.inputs[0]
        target = self.inputs[1]
        aux_input = self.inputs[2] if len(self.inputs) >= 4 else None
        aux_label = self.inputs[3] if len(self.inputs) >= 4 else None
        aux_input_2 = self.inputs[4] if len(self.inputs) >= 5 else None
        aux_label_2 = self.inputs[5] if len(self.inputs) >= 6 else None

        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        if aux_input is not None:
            aux_input.record_stream(torch.cuda.current_stream())
        if aux_label is not None:
            aux_label.record_stream(torch.cuda.current_stream())
        if aux_input_2 is not None:
            aux_input_2.record_stream(torch.cuda.current_stream())
        if aux_label_2 is not None:
            aux_label_2.record_stream(torch.cuda.current_stream())
            
        self.preload()
        return input, target, aux_input, aux_label, aux_input_2, aux_label_2

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)  


if __name__ == "__main__":
    args = parse_args('train')
    print(args)
