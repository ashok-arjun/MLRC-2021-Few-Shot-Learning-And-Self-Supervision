# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import os
identity = lambda x:x
import math

def get_patches(img, transform_jigsaw, transform_patch_jigsaw, permutations):
    if np.random.rand() < 0.30:
        img = img.convert('LA').convert('RGB')## this should be L instead....... need to change that!!

    # import ipdb; ipdb.set_trace()
    # if img.size[0] != 255:
    # img.save('img_ori.png')
    img = transform_jigsaw(img)

    s = float(img.size[0]) / 3
    # s = float(img.shape[1]) / 3
    a = s / 2
    tiles = [None] * 9
    # img.save('img.png')
    for n in range(9):
        i = int(n / 3)
        j = n % 3
        c = [a * i * 2 + a, a * j * 2 + a]
        c = np.array([math.ceil(c[1] - a), math.ceil(c[0] - a), int(c[1] + a ), int(c[0] + a )]).astype(int)
        tile = img.crop(c.tolist())
        # print(c)
        # tile.save(str(n)+'.png')
        # import ipdb; ipdb.set_trace()
        tile = transform_patch_jigsaw(tile)
        # Normalize the patches indipendently to avoid low level features shortcut
        m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
        s[s == 0] = 1
        # import ipdb; ipdb.set_trace()
        norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
        ## Use original normalization
        # norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std =[0.229, 0.224, 0.225])
        tile = norm(tile)
        tiles[n] = tile
        # import torchvision
        # torchvision.utils.save_image(norm(tile),'tile_'+str(n)+'.png')
        # norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std =[0.229, 0.224, 0.225])
        # torchvision.utils.save_image(norm(tile),'tile_patch_'+str(n)+'.png')
    # import ipdb; ipdb.set_trace()

    order = np.random.randint(len(permutations))
    data = [tiles[permutations[order][t]] for t in range(9)]
    data = torch.stack(data, 0)

    return data, int(order)

def retrive_permutations(classes):
    all_perm = np.load('permutations_%d.npy' % (classes))
    # from range [1,9] to [0,8]
    if all_perm.min() == 1:
        all_perm = all_perm - 1

    return all_perm

class SimpleDataset:
    def __init__(self, data_file, transform, transform_depth=None, target_transform=identity, \
                jigsaw=False, transform_jigsaw=None, transform_patch_jigsaw=None, \
                rotation=False, isAircraft=False, grey=False, return_name=False, depth=False):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform
        self.transform_depth = transform_depth

        self.jigsaw = jigsaw
        self.transform_jigsaw = transform_jigsaw
        self.transform_patch_jigsaw = transform_patch_jigsaw
        self.permutations = retrive_permutations(35)

        self.rotation = rotation
        self.isAircraft = False#isAircraft
        self.grey = grey
        self.return_name = return_name
        self.depth = depth

    def __getitem__(self,i):
        # import ipdb; ipdb.set_trace()
        # image_path = os.path.join(self.meta['image_names'][i].replace('images','images_lowres'))
        image_path = os.path.join(self.meta['image_names'][i])
        # image_path = os.path.join(self.meta['image_names'][i].replace('filelists/cars/images/','/data/jcsu/distill-net/data/cars/images_56_224/car_ims/'))
        # print(self.meta['image_names'][i])
        # import ipdb; ipdb.set_trace()
        # image_path = os.path.join(self.meta['image_names'][i].replace('/data/jcsu/CloserLookFewShot/filelists/CUB/CUB_200_2011/images/','/data/jcsu/distill-net/data/cub/images_lowres_full_50_224/'))
        # print(image_path)
        # exit(0)

        if self.grey:
            img = Image.open(image_path).convert('L').convert('RGB')
        else:
            img = Image.open(image_path).convert('RGB')
        
        if self.isAircraft:
            ## crop the banner
            img = img.crop((0,0,img.size[0],img.size[1]-20))

        # i, j, h, w = transforms.RandomResizedCrop.get_params(img,scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
        # img = TF.resized_crop(img, i, j, h, w, (224,224))#self.transform.image_size
        if self.depth:
            image_path_depth = image_path
            image_path_depth = image_path_depth.replace('filelists/CUB/images', \
                        '/mnt/nfs/scratch1/zezhoucheng/MiDaS/out/CUB_200_2011')
            # print(image_path_depth)
            # import ipdb; ipdb.set_trace()
            image_path_depth = image_path_depth.replace('/mnt/nfs/work1/smaji/jcsu/CloserLookFewShot/filelists/miniImagenet/ILSVRC2015/Data/CLS-LOC', \
                        '/mnt/nfs/scratch1/zezhoucheng/MiDaS/out/imagenet')
            image_path_depth = image_path_depth.replace('JPEG', 'png')
            image_path_depth = image_path_depth.replace('jpg', 'png')
            # print(image_path)
            # print(image_path_depth)
            ## jittering still makes it 3 channels, so use 3 channels for now
            ## test time has no
            # img_depth = self.transform_depth(img_depth)
            img_depth = Image.open(image_path_depth).convert('RGB')
            # img_depth = TF.resized_crop(img_depth, i, j, h, w, (224,224))#self.transform.image_size
            # img_depth = TF.to_tensor(img_depth)
            img_depth = self.transform_depth(img_depth)

    
        
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
        target = self.target_transform(self.meta['image_labels'][i])

        if self.jigsaw:
            if self.return_name:
                return img, target, patches, order, image_path
            else:
                return img, target, patches, order
        elif self.rotation:
            if self.return_name:
                return img, target, torch.stack(rotated_imgs, dim=0), rotation_labels, image_path
            else:
                return img, target, torch.stack(rotated_imgs, dim=0), rotation_labels
        elif self.depth:
            # import ipdb; ipdb.set_trace()
            return img, target, img_depth
        else:
            if self.return_name:
                return img, target, image_path
            else:
                return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, data_file, batch_size, transform, transform_depth=None, jigsaw=False, \
                transform_jigsaw=None, transform_patch_jigsaw=None, rotation=False, \
                isAircraft=False, grey=False, depth=False):
        self.jigsaw = jigsaw
        self.transform_jigsaw = transform_jigsaw
        self.transform_patch_jigsaw = transform_patch_jigsaw
        self.rotation = rotation
        self.isAircraft = isAircraft
        self.grey = grey
        self.depth = depth

        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform, transform_depth=transform_depth, jigsaw=self.jigsaw, \
                                    transform_jigsaw=self.transform_jigsaw, transform_patch_jigsaw=self.transform_patch_jigsaw, \
                                    rotation=self.rotation, isAircraft=self.isAircraft, grey=self.grey, depth=self.depth)
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), transform_depth=None, target_transform=identity, \
                jigsaw=False, transform_jigsaw=None, transform_patch_jigsaw=None, rotation=False, \
                isAircraft=False, grey=False, depth=False):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
        self.transform_depth = transform_depth
        # self.transform_depth.transforms[-1] = transforms.Normalize(mean= [0.449] , std=[0.226])

        self.rotation = rotation
        self.isAircraft = False#isAircraft
        self.grey = grey
        self.depth = depth

        self.jigsaw = jigsaw
        if jigsaw:
            self.permutations = retrive_permutations(35)
            self.transform_jigsaw = transform_jigsaw
            self.transform_patch_jigsaw = transform_patch_jigsaw

    def __getitem__(self,i):
        image_path = os.path.join(self.sub_meta[i])
        # image_path = os.path.join(self.sub_meta[i].replace('images','images_lowres'))
        # print(image_path)
        # image_path = os.path.join(self.sub_meta[i].replace('filelists/cars/images/','/data/jcsu/distill-net/data/cars/images_56_224/car_ims/'))
        # image_path = os.path.join(self.sub_meta[i].replace('/data/jcsu/CloserLookFewShot/filelists/CUB/CUB_200_2011/images/','/data/jcsu/distill-net/data/cub/images_lowres_full_50_224/'))
        # print(self.depth)
        # import ipdb; ipdb.set_trace()

        # print(image_path)
        if self.grey:
            img = Image.open(image_path).convert('L').convert('RGB')
        else:
            img = Image.open(image_path).convert('RGB')

        if self.isAircraft:
            ## crop the banner
            img = img.crop((0,0,img.size[0],img.size[1]-20))

        i, j, h, w = transforms.RandomResizedCrop.get_params(img,scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
        img = TF.resized_crop(img, i, j, h, w, (224,224))#self.transform.image_size

        flip = False
        if random.random() < 0.5:
            img = TF.hflip(img)
            flip = True
        img = self.transform(img)

        if self.depth:
        # if True:
            # import ipdb; ipdb.set_trace()
            # filelists/CUB/images/017.Cardinal/Cardinal_0055_18898.jpg
            # /mnt/nfs/scratch1/zezhoucheng/MiDaS/out/CUB_200_2011/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.png
            # import ipdb; ipdb.set_trace()
            # print(image_path)
            image_path_depth = image_path
            image_path_depth = image_path_depth.replace('filelists/CUB/images', \
                        '/mnt/nfs/scratch1/zezhoucheng/MiDaS/out/CUB_200_2011')
            # print(image_path_depth)
            # import ipdb; ipdb.set_trace()
            image_path_depth = image_path_depth.replace('/mnt/nfs/work1/smaji/jcsu/CloserLookFewShot/filelists/miniImagenet/ILSVRC2015/Data/CLS-LOC', \
                        '/mnt/nfs/scratch1/zezhoucheng/MiDaS/out/imagenet')
            image_path_depth = image_path_depth.replace('JPEG', 'png')
            image_path_depth = image_path_depth.replace('jpg', 'png')
            # print(image_path)
            # print(image_path_depth)
            img_depth = Image.open(image_path_depth)
            # img_depth = self.transform_depth(img_depth)## jittering still makes it 3 channels

            img_depth = TF.resized_crop(img_depth, i, j, h, w, (224,224))#self.transform.image_size
            if flip:
                img_depth = TF.hflip(img_depth)
            img_depth = TF.to_tensor(img_depth)
            # import ipdb; ipdb.set_trace()
            # img = torch.mul(img,(img_depth>0.3).float())
            # img_depth = self.transform_depth(img_depth)

            ## added for depth_only
            # image_path = image_path_depth


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

        target = self.target_transform(self.cl)
        
        if self.jigsaw:
            return img, target, patches, order
        elif self.rotation:
            return img, target, torch.stack(rotated_imgs, dim=0), rotation_labels
        elif self.depth:
            # import ipdb; ipdb.set_trace()
            return img, target, img_depth
        else:
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
