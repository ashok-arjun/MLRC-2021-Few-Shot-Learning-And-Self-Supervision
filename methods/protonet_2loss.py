# This code is modified from https://github.com/jakesnell/prototypical-networks 

import models.backbone as backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from models.model_resnet import *
from itertools import cycle

import wandb

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    print("AMP is not installed. If --amp is True, code will fail.")
    
from utils.io_utils import data_prefetcher
from utils.utils import Logger


class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, jigsaw=False, lbda=0.0, rotation=False, tracking=False, lbda_jigsaw=0.0, lbda_rotation=0.0, use_bn=True, pretrain=False, model="resnet18"):
        super(ProtoNet, self).__init__( model_func,  n_way, n_support, use_bn, pretrain, tracking=tracking)
        self.loss_fn = nn.CrossEntropyLoss()

        self.jigsaw = jigsaw
        self.rotation = rotation
        self.lbda = lbda
        self.lbda_jigsaw = lbda_jigsaw
        self.lbda_rotation = lbda_rotation
        # self.lbda_proto = lbda_proto
        self.global_count = 0
        if self.jigsaw and self.rotation:
            self.fc6 = nn.Sequential()
            self.fc6.add_module('fc6_s1',nn.Linear(1024, 1024)) if model != "resnet18" else self.fc6.add_module('fc6_s1',nn.Linear(512, 512))#for resnet
            self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
            self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

            self.fc7_jigsaw = nn.Sequential()
            self.fc7_jigsaw.add_module('fc7',nn.Linear(9*1024,4096)) if model != "resnet18" else self.fc7_jigsaw.add_module('fc7',nn.Linear(9*512,4096))#for resnet
            self.fc7_jigsaw.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7_jigsaw.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier_jigsaw = nn.Sequential()
            self.classifier_jigsaw.add_module('fc8',nn.Linear(4096, 35))

            self.fc7_rotation = nn.Sequential()
            self.fc7_rotation.add_module('fc7',nn.Linear(1024,128)) if model != "resnet18" else self.fc7_rotation.add_module('fc7',nn.Linear(512,128))#for resnet            
            self.fc7_rotation.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7_rotation.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier_rotation = nn.Sequential()
            self.classifier_rotation.add_module('fc8',nn.Linear(128, 4))
        elif self.jigsaw:
            self.fc6 = nn.Sequential()
            self.fc6.add_module('fc6_s1',nn.Linear(1024, 1024)) if model != "resnet18" else self.fc6.add_module('fc6_s1',nn.Linear(512, 512))#for resnet
            self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
            self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

            self.fc7 = nn.Sequential()
            self.fc7.add_module('fc7',nn.Linear(9*1024,4096)) if model != "resnet18" else self.fc7.add_module('fc7',nn.Linear(9*512,4096))#for resnet
            self.fc7.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier = nn.Sequential()
            self.classifier.add_module('fc8',nn.Linear(4096, 35))
        elif self.rotation:
            self.fc6 = nn.Sequential()
            self.fc6.add_module('fc6_s1',nn.Linear(1024, 1024)) if model != "resnet18" else self.fc6.add_module('fc6_s1',nn.Linear(512, 512))#for resnet
            self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
            self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

            self.fc7 = nn.Sequential()
            self.fc7.add_module('fc7',nn.Linear(1024,128)) if model != "resnet18" else self.fc7.add_module('fc7',nn.Linear(512,128))#for resnet            
            self.fc7.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier_rotation = nn.Sequential()
            self.classifier_rotation.add_module('fc8',nn.Linear(128, 4))


    def train_loop(self, epoch, train_loader, optimizer, pbar=None, enable_amp=None, base_loader_u = None, semi_sup=False):
        avg_loss=0
        avg_loss_proto=0
        avg_loss_jigsaw=0
        avg_loss_rotation=0

        if base_loader_u:
            loader = zip(train_loader, cycle(base_loader_u))
        else:
            loader = train_loader

        iter_num = 0 

        for iter_num, inputs in enumerate(loader):
        
            self.global_count += 1

            x = inputs[0] if not base_loader_u else inputs[0][0]

            if semi_sup:
                semi_inputs = x[:, :, 1]
                x = x[:, :, 0]
            else:
                semi_inputs = None

            self.n_query = x.size(1) - self.n_support           
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            # import ipdb; ipdb.set_trace()

            if base_loader_u:
                aux_inputs = inputs[1]
                if len(aux_inputs[2].shape) == 5:
                    aux_inputs[2] = aux_inputs[2].view(self.n_way, self.n_support + self.n_query, *aux_inputs[2].size()[1:])
                if len(aux_inputs) > 4 and len(aux_inputs[4].shape) == 5:
                    aux_inputs[4] = aux_inputs[4].view(self.n_way, self.n_support + self.n_query, *aux_inputs[4].size()[1:])
            else:
                aux_inputs = inputs

            if self.jigsaw and self.rotation:
                loss_proto, loss_jigsaw, loss_rotation, acc, acc_jigsaw, acc_rotation = self.set_forward_loss( x, aux_inputs[2], aux_inputs[3], aux_inputs[4], aux_inputs[5], semi_inputs=semi_inputs )# torch.Size([5, 21, 9, 3, 75, 75]), torch.Size([5, 21])
                loss = (1.0-self.lbda_jigsaw-self.lbda_rotation) * loss_proto + self.lbda_jigsaw * loss_jigsaw + self.lbda_rotation * loss_rotation
                # loss = 0.0 * loss_proto + self.lbda * loss_jigsaw
                Logger.log({'train/loss_proto': float(loss_proto.item())}, step=self.global_count)
                Logger.log({'train/loss_jigsaw': float(loss_jigsaw.item())}, step=self.global_count)
                Logger.log({'train/loss_rotation': float(loss_rotation.item())}, step=self.global_count)
            elif self.jigsaw:
                # import ipdb; ipdb.set_trace()
                loss_proto, loss_jigsaw, acc, acc_jigsaw = self.set_forward_loss( x, aux_inputs[2], aux_inputs[3], semi_inputs=semi_inputs  )# torch.Size([5, 21, 9, 3, 75, 75]), torch.Size([5, 21])
                loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_jigsaw
                # loss = 0.0 * loss_proto + self.lbda * loss_jigsaw
                Logger.log({'train/loss_proto': float(loss_proto.item())}, step=self.global_count)
                Logger.log({'train/loss_jigsaw': float(loss_jigsaw.item())}, step=self.global_count)
            elif self.rotation:
                # import ipdb; ipdb.set_trace()
                loss_proto, loss_rotation, acc, acc_rotation = self.set_forward_loss( x, aux_inputs[2], aux_inputs[3], semi_inputs=semi_inputs  )# torch.Size([5, 21, 9, 3, 75, 75]), torch.Size([5, 21])
                loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_rotation
                Logger.log({'train/loss_proto': float(loss_proto.item())}, step=self.global_count)
                Logger.log({'train/loss_rotation': float(loss_rotation.item())}, step=self.global_count)
            else:
                loss, acc = self.set_forward_loss( x, semi_inputs=semi_inputs  )
                
            if enable_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:   
                loss.backward()
            
            optimizer.step()
            # avg_loss = avg_loss+loss.data[0]
            avg_loss = avg_loss+loss.data
            Logger.log({'train/loss': float(loss.item())}, step=self.global_count)

            pbar.update(1)
            
            if self.jigsaw and self.rotation:
                avg_loss_proto += loss_proto.data
                avg_loss_jigsaw += loss_jigsaw.data
                avg_loss_rotation += loss_rotation.data
                Logger.log({'train/acc_proto': float(acc.item())}, step=self.global_count)
                Logger.log({'train/acc_jigsaw': float(acc_jigsaw.item())}, step=self.global_count)
                Logger.log({'train/acc_rotation': float(acc_rotation.item())}, step=self.global_count)
            elif self.jigsaw:
                avg_loss_proto += loss_proto.data
                avg_loss_jigsaw += loss_jigsaw.data
                Logger.log({'train/acc_proto': float(acc.item())}, step=self.global_count)
                Logger.log({'train/acc_jigsaw': float(acc_jigsaw.item())}, step=self.global_count)
            elif self.rotation:
                avg_loss_proto += loss_proto.data
                avg_loss_rotation += loss_rotation.data
                Logger.log({'train/acc_proto': float(acc.item())}, step=self.global_count)
                Logger.log({'train/acc_rotation': float(acc_rotation.item())}, step=self.global_count)
        
        return avg_loss

    def test_loop(self, test_loader, record = None, base_loader_u=None, semi_sup=False, proto_only=False):
        correct =0
        count = 0
        acc_all = []
        acc_all_jigsaw = []
        acc_all_rotation = []
        
        iter_num = len(test_loader)

        if base_loader_u:
            loader = zip(test_loader, cycle(base_loader_u))
        else:
            loader = test_loader

        i = 0

        for i, inputs in enumerate(loader):

            x = inputs[0] if not base_loader_u else inputs[0][0]

            if semi_sup:
                semi_inputs = x[:, :, 1]
                x = x[:, :, 0]
            else:
                semi_inputs = None

            self.n_query = x.size(1) - self.n_support

            if self.change_way:
                self.n_way  = x.size(0)
                
            if base_loader_u:
                aux_inputs = inputs[1]
                if len(aux_inputs[2].shape) == 5:
                    aux_inputs[2] = aux_inputs[2].view(self.n_way, self.n_support + self.n_query, *aux_inputs[2].size()[1:])
                if len(aux_inputs) > 4 and len(aux_inputs[4].shape) == 5:
                    aux_inputs[4] = aux_inputs[4].view(self.n_way, self.n_support + self.n_query, *aux_inputs[4].size()[1:])
            else:
                aux_inputs = inputs

            if not proto_only:
                if self.jigsaw and self.rotation:
                    correct_this, correct_this_jigsaw, correct_this_rotation, count_this, count_this_jigsaw, count_this_rotation = self.correct(x, aux_inputs[2], aux_inputs[3], aux_inputs[4], aux_inputs[5], semi_inputs=semi_inputs )
                elif self.jigsaw:
                    correct_this, correct_this_jigsaw, count_this, count_this_jigsaw = self.correct(x, aux_inputs[2], aux_inputs[3], semi_inputs=semi_inputs )
                elif self.rotation:
                    correct_this, correct_this_rotation, count_this, count_this_rotation = self.correct(x, aux_inputs[2], aux_inputs[3], semi_inputs=semi_inputs )
                else:
                    correct_this, count_this = self.correct(x, semi_inputs=semi_inputs )
            else:
                correct_this, count_this = self.correct(x, semi_inputs=semi_inputs )
            acc_all.append(correct_this/ count_this*100)

            if not proto_only:
                if self.jigsaw and self.rotation:
                    acc_all_jigsaw.append(correct_this_jigsaw/ count_this_jigsaw*100)
                    acc_all_rotation.append(correct_this_rotation/ count_this_rotation*100)
                elif self.jigsaw:
                    acc_all_jigsaw.append(correct_this_jigsaw/ count_this_jigsaw*100)
                elif self.rotation:
                    acc_all_rotation.append(correct_this_rotation/ count_this_rotation*100)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Protonet Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        if not proto_only:
            if self.jigsaw and self.rotation:
                acc_all_jigsaw  = np.asarray(acc_all_jigsaw)
                acc_mean_jigsaw = np.mean(acc_all_jigsaw)
                acc_std_jigsaw  = np.std(acc_all_jigsaw)
                print('%d Test Jigsaw Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean_jigsaw, 1.96* acc_std_jigsaw/np.sqrt(iter_num)))
                acc_all_rotation  = np.asarray(acc_all_rotation)
                acc_mean_rotation = np.mean(acc_all_rotation)
                acc_std_rotation  = np.std(acc_all_rotation)
                print('%d Test Rotation Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean_rotation, 1.96* acc_std_rotation/np.sqrt(iter_num)))
                return acc_mean, acc_mean_jigsaw, acc_mean_rotation
            elif self.jigsaw:
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
        else:
            return acc_mean, acc_std
    def correct(self, x, patches=None, patches_label=None, patches_rotation=None, patches_label_rotation=None, semi_inputs=None):       
        if self.jigsaw and self.rotation and patches != None and patches_rotation != None:
        	scores, x_, y_, x_rotation, y_rotation = self.set_forward(x,patches=patches,patches_label=patches_label,patches_rotation=patches_rotation, patches_label_rotation=patches_label_rotation, semi_inputs=semi_inputs)
        elif self.jigsaw and patches != None:
            scores, x_, y_ = self.set_forward(x,patches=patches,patches_label=patches_label, semi_inputs=semi_inputs)
        elif self.rotation and patches != None:
            scores, x_, y_ = self.set_forward(x,patches=patches,patches_label=patches_label, semi_inputs=semi_inputs)
        else:
            scores = self.set_forward(x, semi_inputs=semi_inputs)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)

        if self.jigsaw and self.rotation and patches != None and patches_rotation != None:
            pred = torch.max(x_,1)
            top1_correct_jigsaw = torch.sum(pred[1] == y_)
            pred_rotation = torch.max(x_rotation,1)
            top1_correct_rotation = torch.sum(pred_rotation[1] == y_rotation)
            return float(top1_correct), float(top1_correct_jigsaw), float(top1_correct_rotation), len(y_query), len(y_), len(y_rotation)
        elif self.jigsaw and patches != None:
            pred = torch.max(x_,1)
            top1_correct_jigsaw = torch.sum(pred[1] == y_)
            return float(top1_correct), float(top1_correct_jigsaw), len(y_query), len(y_)
        elif self.rotation and patches != None:
            pred = torch.max(x_,1)
            top1_correct_rotation = torch.sum(pred[1] == y_)
            return float(top1_correct), float(top1_correct_rotation), len(y_query), len(y_)
        else:
            return float(top1_correct), len(y_query)

    def set_forward_test(self,x,is_feature = False, semi_inputs=None):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto_nway_kshot     = z_support.view(self.n_way, self.n_support, -1 )
        z_proto = z_proto_nway_kshot.mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)

        if semi_inputs != None:            
            semi_inputs = semi_inputs.cuda()
            semi_inputs = semi_inputs.contiguous().view( self.n_way * (self.n_support + self.n_query), *semi_inputs.size()[2:]) 
            semi_z = self.feature(semi_inputs)
            semi_z = semi_z.view(semi_z.shape[0], -1)
            inner_dist = -euclidean_dist(semi_z, z_proto)
            class_assignments = torch.argmax(F.softmax(inner_dist, dim=1), dim=1)

            z_proto_refined = z_proto_nway_kshot.mean(1)

            for i in range(0, self.n_way):
                class_i_tensors = None
                for j in range(0, semi_z.shape[0]):
                    if class_assignments[j] == i:
                        class_i_tensors = torch.cat([class_i_tensors, semi_z[j]]) if class_i_tensors != None else semi_z[j]
                z_proto_refined[i] = class_i_tensors.mean(0)
                        
            dists = euclidean_dist(z_query, z_proto_refined)
            # get cluster assignments - basic softmax over distance of the prototypes from 
            # recalculate the mean - append them to the corresponding columns in z_proto and then take a mean
            # recal the distance

        scores = -dists
        return scores

    def set_forward(self,x,is_feature = False, patches=None, patches_label=None, patches_rotation=None, patches_label_rotation=None, semi_inputs=None):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto_nway_kshot     = z_support.view(self.n_way, self.n_support, -1 )
        z_proto = z_proto_nway_kshot.mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)

        if semi_inputs != None:            
            semi_inputs = semi_inputs.cuda()
            semi_inputs = semi_inputs.contiguous().view( self.n_way * (self.n_support + self.n_query), *semi_inputs.size()[2:]) 
            semi_z = self.feature(semi_inputs)
            semi_z = semi_z.view(semi_z.shape[0], -1)
            inner_dist = -euclidean_dist(semi_z, z_proto)
            class_assignments = torch.argmax(F.softmax(inner_dist, dim=1), dim=1)

            z_proto_refined = z_proto_nway_kshot.mean(1)

            for i in range(0, self.n_way):
                class_i_tensors = None
                for j in range(0, semi_z.shape[0]):
                    if class_assignments[j] == i:
                        class_i_tensors = torch.cat([class_i_tensors, semi_z[j]]) if class_i_tensors != None else semi_z[j]
                z_proto_refined[i] = class_i_tensors.mean(0)
                        
            dists = euclidean_dist(z_query, z_proto_refined)

        scores = -dists
        
        if self.jigsaw and self.rotation and patches != None and patches_rotation != None:
            # import ipdb; ipdb.set_trace()
            # patches = patches[:,:self.n_support,...]
            # patches = patches[:,:,...]#S is shot+query
            Way,S,T,C,H,W = patches.size()#torch.Size([5, 15, 9, 3, 75, 75])
            B = Way*S
            # patches = patches.contiguous()
            patches = patches.view(Way*S*T,C,H,W).cuda()#torch.Size([675, 3, 64, 64])
            # patches = Variable(patches.cuda())
            # import ipdb; ipdb.set_trace()
            if self.dual_cbam:
                patch_feat = self.feature(patches, jigsaw=True)#torch.Size([675, 512])
            else:
                patch_feat = self.feature(patches)#torch.Size([675, 512])

            x_ = patch_feat.view(Way*S,T,-1)
            x_ = x_.transpose(0,1)#torch.Size([9, 75, 512])

            x_list = []
            for i in range(9):
                # z = self.conv(x_[i])
                # z = self.fc6(z.view(B,-1))
                # import ipdb; ipdb.set_trace()
                z = self.fc6(x_[i])#torch.Size([75, 512])
                z = z.view([B,1,-1])#torch.Size([75, 1, 512])
                x_list.append(z)

            x_ = torch.cat(x_list,1)#torch.Size([75, 9, 512])
            x_ = self.fc7_jigsaw(x_.view(B,-1))#torch.Size([75, 9*512])
            x_ = self.classifier_jigsaw(x_)

            # y_ = patches_label[:,:self.n_support].contiguous().view(-1)
            # y_ = patches_label[:,:].contiguous().view(-1)
            y_ = patches_label.view(-1).cuda()
            # y_ = Variable(y_.cuda())

            ## rotation: ##
        
            # import ipdb; ipdb.set_trace()
            Way,S,T,C,H,W = patches_rotation.size()#torch.Size([5, 21, 4, 3, 224, 224])
            B = Way*S
            patches_rotation = patches_rotation.view(Way*S*T,C,H,W).cuda()
            x_rotation_ = self.feature(patches_rotation)#torch.Size([64, 512, 1, 1])
            x_rotation_ = x_rotation_.squeeze()
            x_rotation_ = self.fc6(x_rotation_)
            x_rotation_ = self.fc7_rotation(x_rotation_)#64,128
            x_rotation_ = self.classifier_rotation(x_rotation_)#64,4
            pred_rotation = torch.max(x_rotation_,1)
            y_rotation_ = patches_label_rotation.view(-1).cuda()

            return scores, x_, y_, x_rotation_, y_rotation_
        elif self.jigsaw and patches != None:
            # import ipdb; ipdb.set_trace()
            # patches = patches[:,:self.n_support,...]
            # patches = patches[:,:,...]#S is shot+query
            Way,S,T,C,H,W = patches.size()#torch.Size([5, 15, 9, 3, 75, 75])
            B = Way*S
            # patches = patches.contiguous()
            patches = patches.view(Way*S*T,C,H,W).cuda()#torch.Size([675, 3, 64, 64])
            # patches = Variable(patches.cuda())
            # import ipdb; ipdb.set_trace()
            if self.dual_cbam:
                patch_feat = self.feature(patches, jigsaw=True)#torch.Size([675, 512])
            else:
                patch_feat = self.feature(patches)#torch.Size([675, 512])

            x_ = patch_feat.view(Way*S,T,-1)
            x_ = x_.transpose(0,1)#torch.Size([9, 75, 512])
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

            return scores, x_, y_
        elif self.rotation and patches != None:
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
            return scores, x_, y_
        else:
            return scores


    def set_forward_loss(self, x, patches=None, patches_label=None, patches_rotation=None, patches_label_rotation=None, semi_inputs=None):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        
        if self.jigsaw and self.rotation and patches != None and patches_rotation != None:
            scores, x_, y_, x_rotation_, y_rotation_ = self.set_forward(x,patches=patches,patches_label=patches_label,patches_rotation=patches_rotation,patches_label_rotation=patches_label_rotation, semi_inputs=semi_inputs)
            pred = torch.max(x_,1)
            acc_jigsaw = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)
            pred_rotation = torch.max(x_rotation_,1)
            acc_rotation = torch.sum(pred_rotation[1] == y_rotation_).cpu().numpy()*1.0/len(y_rotation_)
        elif self.jigsaw and patches != None:
            scores, x_, y_ = self.set_forward(x,patches=patches,patches_label=patches_label, semi_inputs=semi_inputs)
            pred = torch.max(x_,1)
            acc_jigsaw = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)
        elif self.rotation and patches != None:
            scores, x_, y_ = self.set_forward(x,patches=patches,patches_label=patches_label, semi_inputs=semi_inputs)
            pred = torch.max(x_,1)
            acc_rotation = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)
        else:
            scores = self.set_forward(x,patches=patches,patches_label=patches_label, semi_inputs=semi_inputs)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        acc = np.sum(topk_ind[:,0] == y_query.numpy())/len(y_query.numpy())
        y_query = Variable(y_query.cuda())

        if self.jigsaw and self.rotation:
        	return self.loss_fn(scores, y_query), self.loss_fn(x_,y_), self.loss_fn(x_rotation_,y_rotation_), acc, acc_jigsaw, acc_rotation
        elif self.jigsaw:
            return self.loss_fn(scores, y_query), self.loss_fn(x_,y_), acc, acc_jigsaw
        elif self.rotation:
            return self.loss_fn(scores, y_query), self.loss_fn(x_,y_), acc, acc_rotation
        else:
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
