# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import models.backbone as backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

from utils.utils import Logger


try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    print("AMP is not installed. If --amp is True, code will fail.")

class MAML(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, approx = False, jigsaw=False, \
                lbda=0.0, rotation=False, tracking=False, use_bn=True, pretrain=False, model="resnet18", lbda_jigsaw=None, lbda_rotation=None):
        super(MAML, self).__init__(model_func, n_way, n_support, use_bn, pretrain, change_way = False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)
        
        self.n_task     = 1
        self.task_update_num = 5
        self.train_lr = 0.01
        self.approx = approx #first order approx.    

        self.global_count = 0
        self.jigsaw = jigsaw
        self.rotation = rotation
        self.lbda = lbda    
        if self.jigsaw and self.rotation:
            self.fc6 = nn.Sequential()
            self.fc6.add_module('fc6_s1',backbone.Linear_fw(1024, 1024)) if model != "resnet18" else self.fc6.add_module('fc6_s1',backbone.Linear_fw(512, 512))#for resnet
            self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
            self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

            self.fc7_jigsaw = nn.Sequential()
            self.fc7_jigsaw.add_module('fc7',backbone.Linear_fw(9*1024,4096)) if model != "resnet18" else self.fc7_jigsaw.add_module('fc7',backbone.Linear_fw(9*512,4096))#for resnet
            self.fc7_jigsaw.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7_jigsaw.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier_jigsaw = nn.Sequential()
            self.classifier_jigsaw.add_module('fc8',backbone.Linear_fw(4096, 35))

            self.fc7_rotation = nn.Sequential()
            self.fc7_rotation.add_module('fc7',backbone.Linear_fw(1024,128)) if model != "resnet18" else self.fc7_rotation.add_module('fc7',backbone.Linear_fw(512,128))#for resnet            
            self.fc7_rotation.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7_rotation.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier_rotation = nn.Sequential()
            self.classifier_rotation.add_module('fc8',backbone.Linear_fw(128, 4))
        elif self.jigsaw:
            self.fc6 = nn.Sequential()
            self.fc6.add_module('fc6_s1',backbone.Linear_fw(1024, 1024)) if model != "resnet18" else self.fc6.add_module('fc6_s1',backbone.Linear_fw(512, 512))#for resnet
            self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
            self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

            self.fc7 = nn.Sequential()
            self.fc7.add_module('fc7',backbone.Linear_fw(9*1024,4096)) if model != "resnet18" else self.fc7.add_module('fc7',backbone.Linear_fw(9*512,4096))#for resnet
            self.fc7.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier_jigsaw = nn.Sequential()
            self.classifier_jigsaw.add_module('fc8',backbone.Linear_fw(4096, 35))
        elif self.rotation:
            self.fc6 = nn.Sequential()
            self.fc6.add_module('fc6_s1',backbone.Linear_fw(1024, 1024)) if model != "resnet18" else self.fc6.add_module('fc6_s1',backbone.Linear_fw(512, 512))#for resnet
            self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
            self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

            self.fc7 = nn.Sequential()
            self.fc7.add_module('fc7',backbone.Linear_fw(1024,128)) if model != "resnet18" else self.fc7.add_module('fc7',backbone.Linear_fw(512,128))#for resnet            
            self.fc7.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier_rotation = nn.Sequential()
            self.classifier_rotation.add_module('fc8',backbone.Linear_fw(128, 4))
        
    def forward(self,x):
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out.squeeze())
        return scores

    def set_forward(self,x, is_feature = False):
        assert is_feature == False, 'MAML do not support fixed feature' 
        x = x.cuda()
        x_var = Variable(x)
        x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:]) 
        x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:]) 
        y_a_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_support ) )).cuda()
        
        fast_parameters = list(self.parameters())
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        for task_step in range(self.task_update_num):
            scores = self.forward(x_a_i)
            set_loss = self.loss_fn(scores, y_a_i) 
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True, allow_unused=True)#added allow_unused=True
            if self.approx:
                # grad = [ g.detach() for g in grad ]
                # import ipdb; ipdb.set_trace()
                grad_mask = []
                for i,g in enumerate(grad):
                    if g is None:
                        grad_mask.append(i)
                grad = [g.detach() if g is not None else None for g in grad ]
                fast_parameters = []
                for k, weight in enumerate(self.parameters()):
                    if k not in grad_mask:
                        if weight.fast is None:
                            weight.fast = weight - self.train_lr * grad[k] #link fast weight to weight 
                        else:
                            weight.fast = weight.fast - self.train_lr * grad[k]
                        fast_parameters.append(weight.fast)
                    else:
                        fast_parameters.append(weight)

        scores = self.forward(x_b_i)
        return scores

    def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')


    def set_forward_loss(self, x):
        scores = self.set_forward(x, is_feature = False)
        y_b_i = Variable( torch.from_numpy( np.repeat(range( self.n_way ), self.n_query   ) )).cuda()
        loss = self.loss_fn(scores, y_b_i)

        return loss

    def set_forward_loss_unlabel(self, patches=None, patches_label=None):
        if self.jigsaw:
            x_, y_ = self.set_forward_unlabel(patches=patches,patches_label=patches_label)
            pred = torch.max(x_,1)
            acc_jigsaw = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)
        elif self.rotation:
            x_, y_ = self.set_forward_unlabel(patches=patches,patches_label=patches_label)
            pred = torch.max(x_,1)
            acc_rotation = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)

        if self.jigsaw:
            return self.loss_fn(x_,y_), acc_jigsaw
        elif self.rotation:
            return self.loss_fn(x_,y_), acc_rotation

    def set_forward_unlabel(self, patches=None, patches_label=None):
        if len(patches.size()) == 6:
            Way,S,T,C,H,W = patches.size()#torch.Size([5, 15, 9, 3, 75, 75])
            B = Way*S
        elif len(patches.size()) == 5:
            B,T,C,H,W = patches.size()#torch.Size([5, 15, 9, 3, 75, 75])
        if self.jigsaw:
            patches = patches.view(B*T,C,H,W).cuda()#torch.Size([675, 3, 64, 64])
            if self.dual_cbam:
                patch_feat = self.feature(patches, jigsaw=True)#torch.Size([675, 512])
            else:
                patch_feat = self.feature(patches)#torch.Size([675, 512])

            x_ = patch_feat.view(B,T,-1)
            x_ = x_.transpose(0,1)#torch.Size([9, 75, 512])

            x_list = []
            for i in range(9):
                z = self.fc6(x_[i])#torch.Size([75, 512])
                z = z.view([B,1,-1])#torch.Size([75, 1, 512])
                x_list.append(z)

            x_ = torch.cat(x_list,1)#torch.Size([75, 9, 512])
            x_ = self.fc7(x_.view(B,-1))#torch.Size([75, 9*512])
            x_ = self.classifier_jigsaw(x_)

            y_ = patches_label.view(-1).cuda()

            return x_, y_
        elif self.rotation:
            patches = patches.view(B*T,C,H,W).cuda()
            x_ = self.feature(patches)#torch.Size([64, 512, 1, 1])
            x_ = x_.squeeze()
            x_ = self.fc6(x_)
            x_ = self.fc7(x_)#64,128
            x_ = self.classifier_rotation(x_)#64,4
            pred = torch.max(x_,1)
            y_ = patches_label.view(-1).cuda()
            return x_, y_

    def train_loop(self, epoch, train_loader, optimizer, base_loader_u=None, pbar=None, enable_amp=False, semi_sup=False): #overwrite parrent function
        print_freq = 10
        avg_loss=0
        avg_loss_maml=0
        avg_loss_jigsaw=0
        avg_loss_rotation=0
        task_count = 0
        loss_all = []
        optimizer.zero_grad()

        iter_num = 0 

        #train
        for iter_num, inputs in enumerate(train_loader):
            self.global_count += 1
            x = inputs[0]
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"

            loss_maml = self.set_forward_loss(x)
            if self.jigsaw:
                loss_jigsaw, acc_jigsaw = self.set_forward_loss_unlabel(inputs[2], inputs[3])# torch.Size([5, 21, 9, 3, 75, 75]), torch.Size([5, 21])
                loss = (1.0-self.lbda) * loss_maml + self.lbda * loss_jigsaw
                
                Logger.log({'train/loss_maml': float(loss_maml.item())}, step=self.global_count)
                Logger.log({'train/loss_jigsaw': float(loss_jigsaw.item())}, step=self.global_count)
                
                avg_loss_maml += loss_maml.item()
                avg_loss_jigsaw += loss_jigsaw.item()
            elif self.rotation:
                loss_rotation, acc_rotation = self.set_forward_loss_unlabel(inputs[2], inputs[3])# torch.Size([5, 21, 9, 3, 75, 75]), torch.Size([5, 21])
                loss = (1.0-self.lbda) * loss_maml + self.lbda * loss_rotation
                Logger.log({'train/loss_maml': float(loss_maml.item())}, step=self.global_count)
                Logger.log({'train/loss_rotation': float(loss_rotation.item())}, step=self.global_count)
                avg_loss_maml += loss_maml.item()
                avg_loss_rotation += loss_rotation.item()
            else:
                loss = loss_maml
                Logger.log({'train/loss_maml': float(loss_maml.item())}, step=self.global_count)
            avg_loss = avg_loss+loss.item()
            loss_all.append(loss)

            task_count += 1

            if pbar:
                pbar.update(1)

            if task_count == self.n_task:
                loss_q = torch.stack(loss_all).sum(0)
                Logger.log({'train/loss_maml': float(loss_maml.item())}, step=self.global_count)
                if enable_amp:
                    with amp.scale_loss(loss_q, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:   
                    loss_q.backward()

                optimizer.step()
                task_count = 0
                loss_all = []
            optimizer.zero_grad()

        return avg_loss

    def test_loop(self, test_loader, base_loader_u=None, semi_sup=False, proto_only=False, std_also=False): 
        correct =0
        count = 0
        acc_all = []
        acc_all_jigsaw = []
        acc_all_rotation = []
        
        iter_num = len(test_loader) 
        for i, inputs in enumerate(test_loader):
            x = inputs[0]
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0), "MAML do not support way change"
            correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this *100)

            if not proto_only:
                if self.jigsaw:
                    loss_jigsaw, acc_jigsaw = self.set_forward_loss_unlabel(inputs[2], inputs[3])
                    acc_all_jigsaw.append(acc_jigsaw*100)
                elif self.rotation:
                    loss_rotation, acc_rotation = self.set_forward_loss_unlabel(inputs[2], inputs[3])
                    acc_all_rotation.append(acc_rotation*100)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)

        print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        if proto_only or (not self.jigsaw and not self.rotation):
            if std_also:
                return acc_mean, acc_std
            else:
                return acc_mean

        elif self.jigsaw:
            acc_all_jigsaw  = np.asarray(acc_all_jigsaw)
            acc_mean_jigsaw = np.mean(acc_all_jigsaw)
            acc_std_jigsaw  = np.std(acc_all_jigsaw)
            print('%d Test Jigsaw Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean_jigsaw, 1.96* acc_std_jigsaw/np.sqrt(iter_num)))
            return acc_mean, acc_mean_jigsaw
        elif self.rotation:
            acc_all_rotation  = np.asarray(acc_all_rotation)
            acc_mean_rotation = np.mean(acc_all_rotation)
            acc_std_rotation   = np.std(acc_all_rotation)
            print('%d Test Rotation Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean_rotation, 1.96* acc_std_rotation/np.sqrt(iter_num)))
            return acc_mean, acc_mean_rotation