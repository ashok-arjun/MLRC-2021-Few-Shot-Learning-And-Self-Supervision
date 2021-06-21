import socket
import copy
from shutil import copytree
import os
import time
import glob

dset_root = {}
dset_root['CUB'] = '/mnt/nfs/work1/smaji/tsungyulin/bcnn_pytorch/code/dataset/cub'
dset_root['cars'] = '/mnt/nfs/work1/smaji/tsungyulin/bcnn_pytorch/code/dataset/cars'
dset_root['aircrafts'] = '/mnt/nfs/work1/smaji/tsungyulin/bcnn_pytorch/code/dataset/fgvc-aircraft-2013b'
dset_root['inat'] = '/mnt/nfs/work1/smaji/tsungyulin/bcnn_pytorch/code/dataset/inat_2018'
dset_root['flowers'] = '/mnt/nfs/work1/smaji/jcsu/datasets/flowers'
dset_root['dogs'] = '/mnt/nfs/work1/smaji/jcsu/datasets/dogs'

test_code = False 

if 'node' in socket.gethostname() or test_code:
    nfs_dset = copy.deepcopy(dset_root)
    if test_code:
        local_path = os.path.join(os.getenv("HOME"), 'my_local_test')
    else:
        local_path = '/local/jcsu_datasets'
    if not os.path.isdir(local_path):
        time.sleep(1)
        if not os.path.isdir(local_path):
            os.makedirs(local_path)
    for x in dset_root.items():
        folder_name = os.path.basename(x[1])
        dset_root[x[0]] = os.path.join(local_path, folder_name)
        
def wait_dataset_copy_finish(dataset):
    flag_file = os.path.join(dset_root[dataset] + '_flag',
                            'flag_ready.txt')
    while True:
        with open(flag_file, 'r') as f:
            status = f.readline()
        if status == 'True':
            break
        time.sleep(600)


def setup_dataset(dataset):
    if dataset == 'dogs':
        if not os.path.isdir('/local/jcsu_datasets/dogs') or len(glob.glob('/local/jcsu_datasets/dogs/Images/*/*.jpg')) != 20580:
            print('Make dog dir')
            os.system('mkdir /local/jcsu_datasets/dogs')
            # print('Copying dog tar file')
            # os.system('cp /mnt/nfs/work1/smaji/jcsu/datasets/dogs/images.tar /local/jcsu_datasets/dogs')
            print('Untar dog tar file')
            os.system('tar xvf /mnt/nfs/work1/smaji/jcsu/datasets/dogs/images.tar -C /local/jcsu_datasets/dogs')
    elif dataset == 'flowers':
        if not os.path.isdir('/local/jcsu_datasets/flowers') or len(glob.glob('/local/jcsu_datasets/flowers/images/*')) != 8189:
            print('Make flower dir')
            os.system('mkdir /local/jcsu_datasets/flowers')
            # print('Copying flower tar file')
            # os.system('cp /mnt/nfs/work1/smaji/jcsu/datasets/flowers/102flowers.tgz /local/jcsu_datasets/flowers')
            print('Untar flowers tar file')
            os.system('tar xzf /mnt/nfs/work1/smaji/jcsu/datasets/flowers/102flowers.tgz -C /local/jcsu_datasets/flowers')
            os.system('mv /local/jcsu_datasets/flowers/jpg /local/jcsu_datasets/flowers/images')
    elif dataset == 'cars':
        if not os.path.isdir('/local/jcsu_datasets/cars') or len(glob.glob('/local/jcsu_datasets/cars/car_ims/*.jpg')) != 16185:
            print('Make car dir')
            os.system('mkdir /local/jcsu_datasets/cars')
            # print('Copying car tar file')
            # os.system('cp /mnt/nfs/work1/smaji/jcsu/datasets/cars/car_ims.tar.gz /local/jcsu_datasets/cars')
            print('Untar car tar file')
            os.system('tar xzf /mnt/nfs/work1/smaji/jcsu/datasets/cars/car_ims.tar.gz -C /local/jcsu_datasets/cars')
            os.system('mv /local/jcsu_datasets/cars/scratch1/dataset/cars/car_ims /local/jcsu_datasets/cars/')
            os.system('rm -r /local/jcsu_datasets/cars/scratch1')
    elif dataset == 'aircrafts':
        if not os.path.isdir('/local/jcsu_datasets/fgvc-aircraft-2013b') or len(glob.glob('/local/jcsu_datasets/fgvc-aircraft-2013b/data/images/*.jpg')) != 10000:
            print('Make aircraft dir')
            os.system('mkdir -p /local/jcsu_datasets/fgvc-aircraft-2013b/data')
            # print('Copying aircraft tar file')
            # os.system('cp /mnt/nfs/work1/smaji/jcsu/datasets/aircrafts/car_ims.tar.gz /local/jcsu_datasets/fgvc-aircraft-2013b/data')
            print('Untar aircraft tar file')
            os.system('tar xzf /mnt/nfs/work1/smaji/jcsu/datasets/aircrafts/images.tar.gz -C /local/jcsu_datasets/fgvc-aircraft-2013b/data')
            os.system('mv /local/jcsu_datasets/fgvc-aircraft-2013b/data/scratch1/tsungyulin/dataset/fgvc-aircraft-2013b/data/images/ /local/jcsu_datasets/fgvc-aircraft-2013b/data/images')
            os.system('rm -r /local/jcsu_datasets/fgvc-aircraft-2013b/data/scratch1')
    elif dataset == 'CUB':
        if not os.path.isdir('/local/jcsu_datasets/cub') or len(glob.glob('/local/jcsu_datasets/cub/images/*/*.jpg')) != 11788:
            print('Make cub dir')
            os.system('mkdir /local/jcsu_datasets/cub')
            # print('Copying cub tar file')
            # os.system('cp /mnt/nfs/work1/smaji/jcsu/datasets/cub/images.tar.gz /local/jcsu_datasets/cub')
            print('Untar cub tar file')
            os.system('tar xzf /mnt/nfs/work1/smaji/jcsu/datasets/cub/images.tar.gz -C /local/jcsu_datasets/cub')
            os.system('mv /local/jcsu_datasets/cub/scratch1/dataset/cub/images /local/jcsu_datasets/cub/')
            os.system('rm -r /local/jcsu_datasets/cub/scratch1')
    else:
        my_tmp = os.path.join(os.getenv("HOME"), 'tmp')
        if not os.path.isdir(my_tmp):
            os.makedirs(my_tmp)
            os.environ["TMPDIR"] = my_tmp
        if 'node' in socket.gethostname():
            if not os.path.isdir(dset_root[dataset]):
                gypsum_copy_data_to_local(dataset)
            #     if os.path.isdir(os.path.join(dset_root[dataset] + '_flag')):
            #         wait_dataset_copy_finish(dataset)
            #     else:
            #         gypsum_copy_data_to_local(dataset)
            # else:
            #     wait_dataset_copy_finish(dataset)


def gypsum_copy_data_to_local(dataset):
    # flag_file = os.path.join(dset_root[dataset] + '_flag', 'flag_ready.txt')

    # if not os.path.isdir(dset_root[dataset] + '_flag'):
    #     os.makedirs(dset_root[dataset] + '_flag')
    # with open(flag_file, 'w') as f:
    #     f.write('False')
    # if test_code:
    #     import pdb
    #     pdb.set_trace()
    #     pass
    copytree(nfs_dset[dataset], dset_root[dataset])

    # if test_code:
    #     pdb.set_trace()
    #     pass
    # with open(flag_file, 'w') as f:
    #     f.write('True')

