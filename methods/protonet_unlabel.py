# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template_unlabel import MetaTemplate
from model_resnet import *

import wandb

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    print("AMP is not installed. If --amp is True, code will fail.")
    
from io_utils import data_prefetcher


class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, jigsaw=False, lbda=0.0, rotation=False):
        super(ProtoNet, self).__init__( model_func,  n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()

        self.jigsaw = jigsaw
        self.rotation = rotation
        self.lbda = lbda
        # self.lbda_proto = lbda_proto
        self.global_count = 0
        if self.jigsaw:
            self.fc6 = nn.Sequential()
            # self.fc6.add_module('fc6_s1',nn.Linear(1024, 1024))#for conv4
            self.fc6.add_module('fc6_s1',nn.Linear(512, 512))#for resnet
            self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
            self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

            self.fc7 = nn.Sequential()
            # self.fc7.add_module('fc7',nn.Linear(9*1024,4096))#for conv4
            self.fc7.add_module('fc7',nn.Linear(9*512,4096))#for resnet
            self.fc7.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier = nn.Sequential()
            self.classifier.add_module('fc8',nn.Linear(4096, 35))
        if self.rotation:
            self.fc6 = nn.Sequential()
            # self.fc6.add_module('fc6_s1',nn.Linear(1024, 1024))#for conv4
            self.fc6.add_module('fc6_s1',nn.Linear(512, 512))#for resnet
            self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
            self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

            self.fc7 = nn.Sequential()
            # self.fc7.add_module('fc7',nn.Linear(9*1024,4096))#for conv4
            self.fc7.add_module('fc7',nn.Linear(512,128))#for resnet
            self.fc7.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier_rotation = nn.Sequential()
            self.classifier_rotation.add_module('fc8',nn.Linear(128, 4))


    def train_loop(self, epoch, train_loader, optimizer, jigsaw_loader=None, pbar=None, enable_amp=None):
        avg_loss=0
        avg_loss_proto=0
        avg_loss_jigsaw=0
        avg_loss_rotation=0

        iter_num = 0 
        max_iter = len(train_loader)

        # for i, (x,_ ) in enumerate(train_loader):
        while iter_num < max_iter:
            iter_num += 1
            try:
                inputs = iter_loader.next()
            except:
                iter_loader = data_prefetcher(train_loader)
                inputs = iter_loader.next()
            self.global_count += 1
            x = inputs[0]
            self.n_query = x.size(1) - self.n_support           
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()

            loss_proto, acc = self.set_forward_loss( x )
            if self.jigsaw:
                # import ipdb; ipdb.set_trace()
                patches, order = iter(jigsaw_loader).next()
                loss_jigsaw, acc_jigsaw = self.set_forward_loss( patches=patches, patches_label=order )# torch.Size([5, 21, 9, 3, 75, 75]), torch.Size([5, 21])
                loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_jigsaw
                # loss = 0.0 * loss_proto + self.lbda * loss_jigsaw
                wandb.log({'train/loss_proto': float(loss_proto.item())}, step=self.global_count)
                wandb.log({'train/loss_jigsaw': float(loss_jigsaw.item())}, step=self.global_count)
            elif self.rotation:
                # import ipdb; ipdb.set_trace()
                loss_rotation, acc_rotation = self.set_forward_loss( x, inputs[2], inputs[3] )# torch.Size([5, 21, 9, 3, 75, 75]), torch.Size([5, 21])
                loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_rotation
                wandb.log({'train/loss_proto': float(loss_proto.item())}, step=self.global_count)
                wandb.log({'train/loss_rotation': float(loss_rotation.item())}, step=self.global_count)
            else:
                loss = loss_proto
                # loss, acc = self.set_forward_loss( x )

            if enable_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:   
                loss.backward()
                    
            optimizer.step()
            # avg_loss = avg_loss+loss.data[0]
            avg_loss = avg_loss+loss.data
            wandb.log({'train/loss': float(loss.item())}, step=self.global_count)

            pbar.update(1)
            
            if self.jigsaw:
                avg_loss_proto += loss_proto.data
                avg_loss_jigsaw += loss_jigsaw.data
                wandb.log({'train/acc_proto': float(acc.item())}, step=self.global_count)
                wandb.log({'train/acc_jigsaw': float(acc_jigsaw.item())}, step=self.global_count)
            elif self.rotation:
                avg_loss_proto += loss_proto.data
                avg_loss_rotation += loss_rotation.data
                wandb.log({'train/acc_proto': float(acc.item())}, step=self.global_count)
                wandb.log({'train/acc_rotation': float(acc_rotation.item())}, step=self.global_count)
        
        return avg_loss

    def test_loop(self, test_loader, record = None, jigsaw_loader=None):
        correct =0
        count = 0
        acc_all = []
        acc_all_jigsaw = []
        acc_all_rotation = []
        
        iter_num = len(test_loader) 
        i = 0

        while i < iter_num:
            i += 1 
            try:
                inputs = iter_loader.next()
            except:
                iter_loader = data_prefetcher(test_loader)
                inputs = iter_loader.next()
            x = inputs[0]
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)

            if self.jigsaw:
                patches, order = iter(jigsaw_loader).next()
                correct_this_jigsaw, count_this_jigsaw = self.correct(patches=patches, patches_label=order)
            elif self.rotation:
                patches, order = iter(jigsaw_loader).next()
                correct_this_rotation, count_this_rotation = self.correct(patches=patches, patches_label=order)

            correct_this, count_this = self.correct(x=x)
            acc_all.append(correct_this/ count_this*100)
            if self.jigsaw:
                acc_all_jigsaw.append(correct_this_jigsaw/ count_this_jigsaw*100)
            elif self.rotation:
                acc_all_rotation.append(correct_this_rotation/ count_this_rotation*100)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Protonet Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        if self.jigsaw:
            acc_all_jigsaw  = np.asarray(acc_all_jigsaw)
            acc_mean_jigsaw = np.mean(acc_all_jigsaw)
            acc_std_jigsaw  = np.std(acc_all_jigsaw)
            print('%d Test Jigsaw Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean_jigsaw, 1.96* acc_std_jigsaw/np.sqrt(iter_num)))
            return acc_mean, acc_mean_jigsaw
        elif self.rotation:
            acc_all_rotation  = np.asarray(acc_all_rotation)
            acc_mean_rotation = np.mean(acc_all_rotation)
            acc_std_rotation  = np.std(acc_all_rotation)
            print('%d Test Rotation Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean_rotation, 1.96* acc_std_rotation/np.sqrt(iter_num)))
            return acc_mean, acc_mean_rotation
        else:
            return acc_mean

    def correct(self, x=None, patches=None, patches_label=None):       
        if self.jigsaw and x is None:
            x_, y_ = self.set_forward(x,patches=patches,patches_label=patches_label)
            pred = torch.max(x_,1)
            top1_correct_jigsaw = torch.sum(pred[1] == y_)
            return float(top1_correct_jigsaw), len(y_)
        elif self.rotation and x is None:
            x_, y_ = self.set_forward(x,patches=patches,patches_label=patches_label)
            pred = torch.max(x_,1)
            top1_correct_rotation = torch.sum(pred[1] == y_)
            return float(top1_correct_rotation), len(y_)
        elif x is not None:
            scores = self.set_forward(x)
            y_query = np.repeat(range( self.n_way ), self.n_query )
            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:,0] == y_query)
            return float(top1_correct), len(y_query)

    def set_forward(self, x=None, is_feature=False, patches=None, patches_label=None):
        if x is not None:
            z_support, z_query  = self.parse_feature(x,is_feature)

            z_support   = z_support.contiguous()
            z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
            z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

            dists = euclidean_dist(z_query, z_proto)
            scores = -dists
            return scores

        elif self.jigsaw:
            # import ipdb; ipdb.set_trace()
            # patches = patches[:,:self.n_support,...]
            # patches = patches[:,:,...]#S is shot+query
            B,T,C,H,W = patches.size()#torch.Size([32, 9, 3, 64, 64])
            # B = Way*S
            # patches = patches.contiguous()
            patches = patches.view(B*T,C,H,W).cuda()#torch.Size([675, 3, 64, 64])
            # patches = Variable(patches.cuda())
            # import ipdb; ipdb.set_trace()
            if self.dual_cbam:
                patch_feat = self.feature(patches, jigsaw=True)#torch.Size([675, 512])
            else:
                patch_feat = self.feature(patches)#torch.Size([288, 512,1,1])

            x_ = patch_feat.view(B,T,-1)
            x_ = x_.transpose(0,1)#torch.Size([9, 32, 512])

            x_list = []
            for i in range(9):
                # z = self.conv(x_[i])
                # z = self.fc6(z.view(B,-1))
                # import ipdb; ipdb.set_trace()
                z = self.fc6(x_[i])#torch.Size([75, 512])
                z = z.view([B,1,-1])#torch.Size([75, 1, 512])
                x_list.append(z)

            x_ = torch.cat(x_list,1)#torch.Size([75, 9, 512])
            x_ = self.fc7(x_.view(B,-1))#torch.Size([75, 9*512])
            x_ = self.classifier(x_)

            # y_ = patches_label[:,:self.n_support].contiguous().view(-1)
            # y_ = patches_label[:,:].contiguous().view(-1)
            y_ = patches_label.view(-1).cuda()
            # y_ = Variable(y_.cuda())

            return x_, y_
        elif self.rotation:
            # import ipdb; ipdb.set_trace()
            Way,S,T,C,H,W = patches.size()#torch.Size([5, 21, 4, 3, 224, 224])
            B = Way*S
            patches = patches.view(Way*S*T,C,H,W).cuda()
            x_ = self.feature(patches)#torch.Size([64, 512, 1, 1])
            x_ = x_.squeeze()
            x_ = self.fc6(x_)
            x_ = self.fc7(x_)#64,128
            x_ = self.classifier_rotation(x_)#64,4
            pred = torch.max(x_,1)
            y_ = patches_label.view(-1).cuda()
            return x_, y_
        else:
            print('Invalid')
            # return scores


    def set_forward_loss(self, x=None, patches=None, patches_label=None):
        if self.jigsaw and x is None:
            x_, y_ = self.set_forward(x,patches=patches,patches_label=patches_label)
            pred = torch.max(x_,1)
            acc_jigsaw = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)
            return self.loss_fn(x_,y_), acc_jigsaw
        elif self.rotation and x is None:
            x_, y_ = self.set_forward(x,patches=patches,patches_label=patches_label)
            pred = torch.max(x_,1)
            acc_rotation = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)
            return self.loss_fn(x_,y_), acc_rotation
        else:
            y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
            scores = self.set_forward(x,patches=patches,patches_label=patches_label)
            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            acc = np.sum(topk_ind[:,0] == y_query.numpy())/len(y_query.numpy())
            y_query = Variable(y_query.cuda())
            return self.loss_fn(scores, y_query), acc


    def parse_feature(self,x,is_feature):
        x    = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
            z_all       = self.feature(x)
            # import ipdb; ipdb.set_trace()
            # print(z_all.shape)
            z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
            # print(z_all.shape)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        # import ipdb; ipdb.set_trace()
        return z_support, z_query

    


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
