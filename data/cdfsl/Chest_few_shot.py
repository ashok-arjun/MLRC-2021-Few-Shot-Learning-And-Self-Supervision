# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.append("../")
import config.configs as configs

NUM_WORKERS=configs.NUM_WORKERS


class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path=configs.data_dir["ChestX"]+"/sample_labels.csv", \
        image_path = configs.data_dir["ChestX"]):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.img_path = image_path
        self.csv_path = csv_path
        self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]

        self.labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5,  "Pneumothorax": 6}
        
        labels_set = []

        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_name  = []
        self.labels = []


        for name, label in zip(self.image_name_all,self.labels_all):
            label = label.split("|")

            if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[0] in self.used_labels:
                self.labels.append(self.labels_maps[label[0]])
                self.image_name.append(name)
    
        self.data_len = len(self.image_name)

        self.image_name = np.asarray(self.image_name)
        self.labels = np.asarray(self.labels)        

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]

        # Open image
        img_as_img = Image.open(self.img_path +  single_image_name).resize((256, 256)).convert('RGB')
        img_as_img.load()

        # Transform image to tensor
        #img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.labels[index]

        return (img_as_img, single_image_label)

    def __len__(self):
        return self.data_len


identity = lambda x:x

from data.dataset_2loss import retrive_permutations, get_patches

class SimpleDataset:
    def __init__(self, transform, target_transform=identity, \
                jigsaw=False, rotation=False, aug=False):
        self.transform = transform
        self.target_transform = target_transform

        self.jigsaw = jigsaw
        self.permutations = retrive_permutations(35)

        self.rotation = rotation

        self.meta = {}

        self.meta['image_names'] = []
        self.meta['image_labels'] = []

        # d = CustomDatasetFromImages()

        csv_path=configs.data_dir["ChestX"]+"/sample_labels.csv"
        self.image_path = configs.data_dir["ChestX"]

        self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]

        self.labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5,  "Pneumothorax": 6}
        
        labels_set = []

        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_name  = []
        self.labels = []


        for name, label in zip(self.image_name_all,self.labels_all):
            label = label.split("|")

            if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[0] in self.used_labels:
                self.labels.append(self.labels_maps[label[0]])
                self.image_name.append(name)
    
        self.data_len = len(self.image_name)

        self.image_name = np.asarray(self.image_name)
        self.labels = np.asarray(self.labels)  

        self.meta['image_names'] = self.image_name 
        self.meta['image_labels'] = self.labels

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

    def __getitem__(self, i):

        img = Image.open(os.path.join(self.image_path, self.meta['image_names'][i])).convert('RGB')
        target = self.meta['image_labels'][i] 

        if self.jigsaw:
            patches, order = get_patches(img, self.transform_jigsaw, self.transform_patch_jigsaw, self.permutations)
        if self.rotation:
            rotated_imgs = [
                    self.transform(img),
                    self.transform(img.rotate(90,expand=True)),
                    self.transform(img.rotate(180,expand=True)),
                    self.transform(img.rotate(270,expand=True))
                ]
            rotation_labels = torch.LongTensor([0, 1, 2, 3])

        img = self.transform(img)
        target = self.target_transform(target)

        if self.jigsaw:
            return img, target, patches, order
        elif self.rotation:
            return img, target, torch.stack(rotated_imgs, dim=0), rotation_labels
        else:
            return img, target


    def rgb_jittering(self, im):
        im = np.array(im, 'int32')
        for ch in range(3):
            im[:, :, ch] += np.random.randint(-2, 2)
        im[im > 255] = 255
        im[im < 0] = 0
        return im.astype('uint8')

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, batch_size, transform):

        self.sub_meta = {}
        self.cl_list = range(7)


        for cl in self.cl_list:
            self.sub_meta[cl] = []


        self.meta = {}

        self.meta['image_names'] = []
        self.meta['image_labels'] = []

        # d = CustomDatasetFromImages()

        csv_path=configs.data_dir["ChestX"]+"/sample_labels.csv"
        self.image_path = configs.data_dir["ChestX"]

        self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]

        self.labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5,  "Pneumothorax": 6}
        
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_name  = []
        self.labels = []

        for name, label in zip(self.image_name_all,self.labels_all):
            label = label.split("|")

            if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[0] in self.used_labels:
                self.labels.append(self.labels_maps[label[0]])
                self.image_name.append(name)
    
        self.data_len = len(self.image_name)

        self.image_name = np.asarray(self.image_name)
        self.labels = np.asarray(self.labels)  

        self.meta['image_names'] = self.image_name 
        self.meta['image_labels'] = self.labels

        for i, (label) in enumerate(self.meta["image_labels"]):
            self.sub_meta[label].append(os.path.join(self.image_path, self.meta['image_names'][i]))

        # for key, item in self.sub_meta.items():
        #     print (len(self.sub_meta[key]))
    
        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
       
        for cl in self.cl_list:
            # print (cl)
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):

        img = self.transform(Image.open(self.sub_meta[i]).convert("RGB"))
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)

        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 

        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':

            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 

class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size,  \
                jigsaw=False, rotation=False):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

        self.jigsaw = jigsaw
        self.permutations = retrive_permutations(35)

        self.rotation = rotation
        

    def get_data_loader(self, aug, drop_last=False): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(transform, jigsaw=self.jigsaw, rotation=self.rotation, aug=aug)

        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = NUM_WORKERS, pin_memory = True, drop_last=drop_last)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way=5, n_support=5, n_query=16, n_eposide = 100):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = NUM_WORKERS, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

if __name__ == '__main__':

    base_datamgr            = SetDataManager(224, n_query = 16, n_support = 5)
    base_loader             = base_datamgr.get_data_loader(aug = True)