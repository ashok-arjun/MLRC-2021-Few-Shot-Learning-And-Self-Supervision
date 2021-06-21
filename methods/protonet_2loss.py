# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from model_resnet import *

class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, jigsaw=False, lbda=0.0, rotation=False, tracking=False, lbda_jigsaw=0.0, lbda_rotation=0.0, use_bn=True):
        super(ProtoNet, self).__init__( model_func,  n_way, n_support, use_bn)
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
            # self.fc6.add_module('fc6_s1',nn.Linear(1024, 1024))#for conv4
            self.fc6.add_module('fc6_s1',nn.Linear(512, 512))#for resnet
            self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
            self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

            self.fc7_jigsaw = nn.Sequential()
            # self.fc7.add_module('fc7',nn.Linear(9*1024,4096))#for conv4
            self.fc7_jigsaw.add_module('fc7',nn.Linear(9*512,4096))#for resnet
            self.fc7_jigsaw.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7_jigsaw.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier_jigsaw = nn.Sequential()
            self.classifier_jigsaw.add_module('fc8',nn.Linear(4096, 35))

            self.fc7_rotation = nn.Sequential()
            # self.fc7.add_module('fc7',nn.Linear(9*1024,4096))#for conv4
            self.fc7_rotation.add_module('fc7',nn.Linear(512,128))#for resnet
            self.fc7_rotation.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7_rotation.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier_rotation = nn.Sequential()
            self.classifier_rotation.add_module('fc8',nn.Linear(128, 4))
        elif self.jigsaw:
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
        elif self.rotation:
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


    def train_loop(self, epoch, train_loader, optimizer, writer):
        print_freq = 10

        avg_loss=0
        avg_loss_proto=0
        avg_loss_jigsaw=0
        avg_loss_rotation=0
        # for i, (x,_ ) in enumerate(train_loader):
        for i, inputs in enumerate(train_loader):
            self.global_count += 1
            x = inputs[0]
            # import ipdb; ipdb.set_trace()
            self.n_query = x.size(1) - self.n_support           
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            # import ipdb; ipdb.set_trace()
            if self.jigsaw and self.rotation:
                loss_proto, loss_jigsaw, loss_rotation, acc, acc_jigsaw, acc_rotation = self.set_forward_loss( x, inputs[2], inputs[3], inputs[4], inputs[5] )# torch.Size([5, 21, 9, 3, 75, 75]), torch.Size([5, 21])
                loss = (1.0-self.lbda_jigsaw-self.lbda_rotation) * loss_proto + self.lbda_jigsaw * loss_jigsaw + self.lbda_rotation * loss_rotation
                # loss = 0.0 * loss_proto + self.lbda * loss_jigsaw
                writer.add_scalar('train/loss_proto', float(loss_proto.data.item()), self.global_count)
                writer.add_scalar('train/loss_jigsaw', float(loss_jigsaw.data.item()), self.global_count)
                writer.add_scalar('train/loss_rotation', float(loss_rotation.data.item()), self.global_count)
            elif self.jigsaw:
                # import ipdb; ipdb.set_trace()
                loss_proto, loss_jigsaw, acc, acc_jigsaw = self.set_forward_loss( x, inputs[2], inputs[3] )# torch.Size([5, 21, 9, 3, 75, 75]), torch.Size([5, 21])
                loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_jigsaw
                # loss = 0.0 * loss_proto + self.lbda * loss_jigsaw
                writer.add_scalar('train/loss_proto', float(loss_proto.data.item()), self.global_count)
                writer.add_scalar('train/loss_jigsaw', float(loss_jigsaw.data.item()), self.global_count)
            elif self.rotation:
                # import ipdb; ipdb.set_trace()
                loss_proto, loss_rotation, acc, acc_rotation = self.set_forward_loss( x, inputs[2], inputs[3] )# torch.Size([5, 21, 9, 3, 75, 75]), torch.Size([5, 21])
                loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_rotation
                writer.add_scalar('train/loss_proto', float(loss_proto.data.item()), self.global_count)
                writer.add_scalar('train/loss_rotation', float(loss_rotation.data.item()), self.global_count)
            else:
                loss, acc = self.set_forward_loss( x )
            loss.backward()
            optimizer.step()
            # avg_loss = avg_loss+loss.data[0]
            avg_loss = avg_loss+loss.data
            writer.add_scalar('train/loss', float(loss.data.item()), self.global_count)

            if self.jigsaw and self.rotation:
                avg_loss_proto += loss_proto.data
                avg_loss_jigsaw += loss_jigsaw.data
                avg_loss_rotation += loss_rotation.data
                writer.add_scalar('train/acc_proto', acc, self.global_count)
                writer.add_scalar('train/acc_jigsaw', acc_jigsaw, self.global_count)
                writer.add_scalar('train/acc_rotation', acc_rotation, self.global_count)
            elif self.jigsaw:
                avg_loss_proto += loss_proto.data
                avg_loss_jigsaw += loss_jigsaw.data
                writer.add_scalar('train/acc_proto', acc, self.global_count)
                writer.add_scalar('train/acc_jigsaw', acc_jigsaw, self.global_count)
            elif self.rotation:
                avg_loss_proto += loss_proto.data
                avg_loss_rotation += loss_rotation.data
                writer.add_scalar('train/acc_proto', acc, self.global_count)
                writer.add_scalar('train/acc_rotation', acc_rotation, self.global_count)

            if (i+1) % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                if self.jigsaw and self.rotation:
                	print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Proto {:f} | Loss Jigsaw {:f} | Loss Rotation {:f}'.\
                        format(epoch, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_proto/float(i+1), avg_loss_jigsaw/float(i+1), avg_loss_rotation/float(i+1)))
                elif self.jigsaw:
                    print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Proto {:f} | Loss Jigsaw {:f}'.\
                        format(epoch, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_proto/float(i+1), avg_loss_jigsaw/float(i+1)))
                elif self.rotation:
                    print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Proto {:f} | Loss Rotation {:f}'.\
                        format(epoch, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_proto/float(i+1), avg_loss_rotation/float(i+1)))
                else:
                    print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i+1, len(train_loader), avg_loss/float(i+1)))

    def test_loop(self, test_loader, record = None):
        correct =0
        count = 0
        acc_all = []
        acc_all_jigsaw = []
        acc_all_rotation = []
        
        iter_num = len(test_loader) 
        for i, inputs in enumerate(test_loader):
            x = inputs[0]
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)

            if self.jigsaw and self.rotation:
            	correct_this, correct_this_jigsaw, correct_this_rotation, count_this, count_this_jigsaw, count_this_rotation = self.correct(x, inputs[2], inputs[3], inputs[4], inputs[5])
            elif self.jigsaw:
                correct_this, correct_this_jigsaw, count_this, count_this_jigsaw = self.correct(x, inputs[2], inputs[3])
            elif self.rotation:
                correct_this, correct_this_rotation, count_this, count_this_rotation = self.correct(x, inputs[2], inputs[3])
            else:
                correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this*100)
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

    def correct(self, x, patches=None, patches_label=None, patches_rotation=None, patches_label_rotation=None):       
        if self.jigsaw and self.rotation:
        	scores, x_, y_, x_rotation, y_rotation = self.set_forward(x,patches=patches,patches_label=patches_label,patches_rotation=patches_rotation, patches_label_rotation=patches_label_rotation)
        elif self.jigsaw:
            scores, x_, y_ = self.set_forward(x,patches=patches,patches_label=patches_label)
        elif self.rotation:
            scores, x_, y_ = self.set_forward(x,patches=patches,patches_label=patches_label)
        else:
            scores = self.set_forward(x)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)

        if self.jigsaw and self.rotation:
            pred = torch.max(x_,1)
            top1_correct_jigsaw = torch.sum(pred[1] == y_)
            pred_rotation = torch.max(x_rotation,1)
            top1_correct_rotation = torch.sum(pred_rotation[1] == y_rotation)
            return float(top1_correct), float(top1_correct_jigsaw), float(top1_correct_rotation), len(y_query), len(y_), len(y_rotation)
        elif self.jigsaw:
            pred = torch.max(x_,1)
            top1_correct_jigsaw = torch.sum(pred[1] == y_)
            return float(top1_correct), float(top1_correct_jigsaw), len(y_query), len(y_)
        elif self.rotation:
            pred = torch.max(x_,1)
            top1_correct_rotation = torch.sum(pred[1] == y_)
            return float(top1_correct), float(top1_correct_rotation), len(y_query), len(y_)
        else:
            return float(top1_correct), len(y_query)

    def set_forward(self,x,is_feature = False, patches=None, patches_label=None, patches_rotation=None, patches_label_rotation=None):
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists

        if self.jigsaw and self.rotation:
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
        elif self.jigsaw:
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
            return scores, x_, y_
        else:
            return scores


    def set_forward_loss(self, x, patches=None, patches_label=None, patches_rotation=None, patches_label_rotation=None):
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        
        if self.jigsaw and self.rotation:
            scores, x_, y_, x_rotation_, y_rotation_ = self.set_forward(x,patches=patches,patches_label=patches_label,patches_rotation=patches_rotation,patches_label_rotation=patches_label_rotation)
            pred = torch.max(x_,1)
            acc_jigsaw = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)
            pred_rotation = torch.max(x_rotation_,1)
            acc_rotation = torch.sum(pred_rotation[1] == y_rotation_).cpu().numpy()*1.0/len(y_rotation_)
        elif self.jigsaw:
            scores, x_, y_ = self.set_forward(x,patches=patches,patches_label=patches_label)
            pred = torch.max(x_,1)
            acc_jigsaw = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)
        elif self.rotation:
            scores, x_, y_ = self.set_forward(x,patches=patches,patches_label=patches_label)
            pred = torch.max(x_,1)
            acc_rotation = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)
        else:
            scores = self.set_forward(x,patches=patches,patches_label=patches_label)

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
