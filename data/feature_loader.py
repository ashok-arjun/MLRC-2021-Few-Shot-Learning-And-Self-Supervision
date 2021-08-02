import torch
import numpy as np
import h5py

class SimpleHDF5Dataset:
    def __init__(self, file_handle = None, depth=False):
        self.depth = depth
        if file_handle == None:
            self.f = ''
            self.all_feats_dset = []
            self.all_labels = []
            self.total = 0 
        else:
            self.f = file_handle
            self.all_feats_dset = self.f['all_feats'][...]
            self.all_labels = self.f['all_labels'][...]
            # if depth:
                # self.all_feats_depth = self.f['all_feats_depth'][...]
            self.total = self.f['count'][0]
           # print('here')
    def __getitem__(self, i):
        if self.depth:
            return torch.Tensor(self.all_feats_dset[i,:]), torch.Tensor(self.all_feats_depth[i,:]), int(self.all_labels[i])
        else:    
            return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i])

    def __len__(self):
        return self.total

def init_loader(filename, depth=False):
    with h5py.File(filename, 'r') as f:
        fileset = SimpleHDF5Dataset(f, depth=depth)
    #labels = [ l for l  in fileset.all_labels if l != 0]
    feats = fileset.all_feats_dset
    labels = fileset.all_labels
    # if depth:
    #     feats_depth = fileset.all_feats_depth

    while np.sum(feats[-1]) == 0:
        feats  = np.delete(feats,-1,axis = 0)
        labels = np.delete(labels,-1,axis = 0)
        # if depth:
        #     feats_depth = np.delete(feats_depth,-1,axis = 0)
        
    class_list = np.unique(np.array(labels)).tolist() 
    inds = range(len(labels))

    cl_data_file = {}
    for cl in class_list:
        cl_data_file[cl] = []
    for ind in inds:
        # if depth:
        #     cl_data_file[labels[ind]].append((feats[ind],feats_depth[ind]))
        # else:
        cl_data_file[labels[ind]].append(feats[ind])

    return cl_data_file
