# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from data.dataset_depth import SimpleDataset, SetDataset, EpisodicBatchSampler
from abc import abstractmethod

NUM_WORKERS=12

class TransformLoader:
    def __init__(self, image_size, depth,
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
        self.depth = depth
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            if self.depth:
                method = transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0)
            else:
                method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomResizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        # elif transform_type=='Scale':
        #     return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Resize':
            return method(int(self.image_size))
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug=False, depth=False):
        if aug:
            if depth:
                # transform_list = ['ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
                # transform_list = ['ImageJitter', 'RandomHorizontalFlip', 'ToTensor']
                ## flip is written in dataset now
                transform_list = ['ImageJitter', 'ToTensor']
            else:
                # transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
                ## flip is written in dataset now
                transform_list = ['RandomResizedCrop', 'ImageJitter', 'ToTensor', 'Normalize']
        else:
            # transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']
            if depth:
                transform_list = ['Resize','CenterCrop', 'ToTensor']
            else:
                transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size, jigsaw=False, rotation=False, isAircraft=False, grey=False, \
                return_name=False, drop_last=False, shuffle=True, depth=False):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        if grey:
            self.trans_loader = TransformLoader(image_size, depth, normalize_param = dict(mean= [0.449, 0.449, 0.449] , std=[0.226, 0.226, 0.226]))
        else:
            self.trans_loader = TransformLoader(image_size, depth)

        self.jigsaw = jigsaw
        self.rotation = rotation
        self.isAircraft = isAircraft
        self.grey = grey
        self.return_name = return_name
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.depth = depth

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)

        transform_depth = None
        if self.depth:
            transform_depth = self.trans_loader.get_composed_transform(aug, depth=True)

        ## Add transform for jigsaw puzzle
        self.transform_patch_jigsaw = None
        self.transform_jigsaw = None
        if self.jigsaw:
            if aug:
                self.transform_jigsaw = transforms.Compose([
                    # transforms.Resize(256),
                    # transforms.CenterCrop(225),
                    ## follow paper setting:
                    # transforms.Resize(255),
                    # transforms.CenterCrop(240),
                    ## setting of my experiment before 0515
                    transforms.RandomResizedCrop(255,scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip()])
                    # transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    #                      std =[0.229, 0.224, 0.225])])
            else:
                self.transform_jigsaw = transforms.Compose([
                    # transforms.Resize(256),
                    # transforms.CenterCrop(225),])
                    transforms.RandomResizedCrop(225,scale=(0.5, 1.0))])
                    # transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    #                      std =[0.229, 0.224, 0.225])])
            self.transform_patch_jigsaw = transforms.Compose([
                transforms.RandomCrop(64),
                # transforms.Resize((75, 75), Image.BILINEAR),
                transforms.Lambda(self.rgb_jittering),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                # std =[0.229, 0.224, 0.225])
            ])

        dataset = SimpleDataset(data_file, transform, transform_depth=transform_depth, jigsaw=self.jigsaw, \
                    transform_jigsaw=self.transform_jigsaw, transform_patch_jigsaw=self.transform_patch_jigsaw, \
                    rotation=self.rotation, isAircraft=self.isAircraft, grey=self.grey, return_name=self.return_name, depth=self.depth)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = self.shuffle, num_workers = NUM_WORKERS, pin_memory = True, drop_last=self.drop_last)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

    def rgb_jittering(self, im):
        im = np.array(im, 'int32')
        for ch in range(3):
            im[:, :, ch] += np.random.randint(-2, 2)
        im[im > 255] = 255
        im[im < 0] = 0
        return im.astype('uint8')

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, n_eposide =100, \
                jigsaw=False, lbda=0.0, lbda_proto=0.0, rotation=False, isAircraft=False, grey=False, depth=False):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        if grey:
            self.trans_loader = TransformLoader(image_size, depth, normalize_param = dict(mean= [0.449, 0.449, 0.449] , std=[0.226, 0.226, 0.226]))
        else:
            self.trans_loader = TransformLoader(image_size, depth)

        self.jigsaw = jigsaw
        self.rotation = rotation
        self.isAircraft = isAircraft
        self.grey = grey
        self.depth = depth

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)

        transform_depth = None
        if self.depth:
            transform_depth = self.trans_loader.get_composed_transform(aug, depth=True)

        ## Add transform for jigsaw puzzle
        self.transform_patch_jigsaw = None
        self.transform_jigsaw = None
        if self.jigsaw:
            if aug:
                self.transform_jigsaw = transforms.Compose([
                    # transforms.Resize(256),
                    # transforms.CenterCrop(225),
                    ## follow paper setting:
                    # transforms.Resize(255),
                    # transforms.CenterCrop(240),
                    ## setting of my experiment before 0515
                    transforms.RandomResizedCrop(255,scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip()])
                    # transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    #                      std =[0.229, 0.224, 0.225])])
            else:
                self.transform_jigsaw = transforms.Compose([
                    # transforms.Resize(256),
                    # transforms.CenterCrop(225),])
                    transforms.RandomResizedCrop(225,scale=(0.5, 1.0))])
                    # transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    #                      std =[0.229, 0.224, 0.225])])
            self.transform_patch_jigsaw = transforms.Compose([
                transforms.RandomCrop(64),
                # transforms.Resize((75, 75), Image.BILINEAR),
                transforms.Lambda(self.rgb_jittering),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                # std =[0.229, 0.224, 0.225])
            ])

        dataset = SetDataset(data_file , self.batch_size, transform, transform_depth=transform_depth, jigsaw=self.jigsaw, \
                            transform_jigsaw=self.transform_jigsaw, transform_patch_jigsaw=self.transform_patch_jigsaw, \
                            rotation=self.rotation, isAircraft=self.isAircraft, grey=self.grey, depth=self.depth)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = NUM_WORKERS, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

    def rgb_jittering(self, im):
        im = np.array(im, 'int32')
        for ch in range(3):
            im[:, :, ch] += np.random.randint(-2, 2)
        im[im > 255] = 255
        im[im < 0] = 0
        return im.astype('uint8')

