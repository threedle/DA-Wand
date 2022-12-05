import torch.utils.data as data
import numpy as np
import pickle
import os

class BaseDataset(data.Dataset):

    def __init__(self, opt):
        self.opt = opt
        self.mean = 0
        self.std = 1
        self.ninput_channels = None
        self.dataset = data.Dataset
        super(BaseDataset, self).__init__()

    def vertex_mean_std(self):
        """ Computes Mean and Standard Deviation from Training Data
        If mean/std file doesn't exist, will compute one
        :returns
        mean: N-dimensional mean
        std: N-dimensional standard deviation
        ninput_channels: N

        NOTE: Run this once before training to cache all the computed mesh features 
        """
        # Save mean/std data based on training set name 
        self.ninput_vertices = 0
        mean_std_cache = os.path.join(self.dir, f'mean_std_{self.opt.name}_cache.p')
        if not os.path.isfile(mean_std_cache) or self.opt.overwritemeanstd == True:
            print(f'computing mean std from {self.opt.phase} data...')
            bad_indices = [] 
            self.mean, self.std = np.array(0), np.array(1)
            mean, std = np.array(0), np.array(0)
            max_nvertices = 0
            max_channels = 0
            if self.opt.time: 
                import time 
                t0 = time.time() 
            for i, data in enumerate(self):
                if i % 500 == 0:
                    print('{} of {}'.format(i, self.size))
                # Store bad indices and update the dataset after this loop  
                if data is None:
                    bad_indices.append(i)
                    continue 
                features = data['vertexfeatures']
                mean = mean + features.mean(axis=0)
                std = std + features.std(axis=0)
                
                if max_nvertices < features.shape[0]:
                    max_nvertices = features.shape[0]
                    
                if max_channels < features.shape[1]:
                    max_channels = features.shape[1]
            if self.opt.time: 
                print(f"Data preprocessing: {time.time() - t0:0.5f} seconds")
                
            if len(mean.shape) > 0:
                mean /= (i+1 - len(bad_indices))
                std /= (i+1 - len(bad_indices))
                # Prevent degenerate normalization 
                mean[mean <= 1e-5] = 0
                std[std <= 1e-5] = 1
            else:
                mean = np.array([0])
                std = np.array([1])
            transform_dict = {'ninput_channels': max_channels, 'max_nvertices': max_nvertices, 'mean': mean, 'std': std}
            with open(mean_std_cache, 'wb') as f:
                pickle.dump(transform_dict, f)
            print('saved: ', mean_std_cache)
            
            # Only need to overwrite once
            self.opt.overwritemeanstd = False
            
            # Fix bad indices 
            if len(bad_indices) > 0:
                print(f"Removing {len(bad_indices)} bad samples from dataset ... ")
                self.paths = [path for i, path in enumerate(self.paths) if i not in bad_indices]
                # NOTE: will be aug specific if test 
                self.cachepaths = [path for i, path in enumerate(self.cachepaths) if i not in bad_indices]
                self.anchorcachepaths = [path for i, path in enumerate(self.anchorcachepaths) if i not in bad_indices]
                self.labelpaths = [path for i, path in enumerate(self.labelpaths) if i not in bad_indices]
                self.augs = [aug for i, aug in enumerate(self.augs) if i not in bad_indices]
                self.anchor_fs = [anchor for i, anchor in enumerate(self.anchor_fs) if i not in bad_indices]
                
                # Reindex the internal data dictionary as well 
                mapping = np.zeros(self.size).astype(int)
                good_indices = np.sort(list(set(range(self.size)).difference(set(bad_indices))))
                mapping[good_indices] = np.arange(len(good_indices))
                new_data = {mapping[i]: d for i, d in sorted(self.data.items()) if i in good_indices}
                self.data = new_data
                
                # Make sure everything is set now 
                for d in self.data.values():
                    assert isinstance(d, dict)
                
                self.size = len(self.paths)
                
            # Only need to overwrite cache once 
            self.opt.overwritecache = False 
                
        # open mean / std from file
        with open(mean_std_cache, 'rb') as f:
            transform_dict = pickle.load(f)
            print('Loaded mean/std, # channels, # vertices from cache.')
            self.mean = transform_dict['mean']
            self.std = transform_dict['std']
            self.ninput_channels = transform_dict['ninput_channels']
            self.ninput_vertices = transform_dict['max_nvertices']
            
    def get_mean_std(self):
        """ Computes Mean and Standard Deviation from Training Data
        If mean/std file doesn't exist, will compute one
        :returns
        mean: N-dimensional mean
        std: N-dimensional standard deviation
        ninput_channels: N
        (here N=5)

        NOTE: Run this once before training to cache all the computed mesh features 
        """
        # Save mean/std data based on training set name 
        mean_std_cache = os.path.join(self.dir, f'mean_std_{self.opt.name}_cache.p')
        if not os.path.isfile(mean_std_cache) or self.opt.overwritemeanstd == True:
            print(f'computing mean std from {self.opt.phase} data...')
            bad_indices = [] 
            self.mean, self.std = np.array(0), np.array(1)
            mean, std = np.array(0), np.array(0)
            max_nedges = 0 # Set input edges to max edge mesh among files
            max_channels = 0
            if self.opt.time: 
                import time 
                t0 = time.time() 
            for i, data in enumerate(self):
                if i % 500 == 0:
                    print('{} of {}'.format(i, self.size))
                # Store bad indices and update the dataset after this loop  
                if data is None:
                    bad_indices.append(i)
                    continue 
                features = data['edge_features']
                mean = mean + features.mean(axis=1)
                std = std + features.std(axis=1)
                
                if max_nedges < features.shape[1]:
                    max_nedges = features.shape[1]
                    
                if max_channels < features.shape[0]:
                    max_channels = features.shape[0]
            if self.opt.time: 
                print(f"Data preprocessing: {time.time() - t0:0.5f} seconds")
                
            if len(mean.shape) > 0:
                mean /= (i+1 - len(bad_indices))
                std /= (i+1 - len(bad_indices))
                # Prevent degenerate normalization 
                mean[mean <= 1e-5] = 0
                std[std <= 1e-5] = 1
            else:
                mean = np.array([0])
                std = np.array([1])
            transform_dict = {'ninput_channels': max_channels, 'max_edges': max_nedges, 'mean': mean, 'std': std}
            with open(mean_std_cache, 'wb') as f:
                pickle.dump(transform_dict, f)
            print('saved: ', mean_std_cache)
            
            # Only need to overwrite once
            self.opt.overwritemeanstd = False
            
            # Fix bad indices 
            if len(bad_indices) > 0:
                print(f"Removing {len(bad_indices)} bad samples from dataset ... ")
                self.paths = [path for i, path in enumerate(self.paths) if i not in bad_indices]
                # NOTE: will be aug specific if test 
                self.cachepaths = [path for i, path in enumerate(self.cachepaths) if i not in bad_indices]
                self.anchorcachepaths = [path for i, path in enumerate(self.anchorcachepaths) if i not in bad_indices]
                self.labelpaths = [path for i, path in enumerate(self.labelpaths) if i not in bad_indices]
                self.augs = [aug for i, aug in enumerate(self.augs) if i not in bad_indices]
                self.anchor_fs = [anchor for i, anchor in enumerate(self.anchor_fs) if i not in bad_indices]
                
                # Reindex the internal data dictionary as well 
                mapping = np.zeros(self.size).astype(int)
                good_indices = np.sort(list(set(range(self.size)).difference(set(bad_indices))))
                mapping[good_indices] = np.arange(len(good_indices))
                new_data = {mapping[i]: d for i, d in sorted(self.data.items()) if i in good_indices}
                self.data = new_data
                
                # Make sure everything is set now 
                for d in self.data.values():
                    assert isinstance(d, dict)
                
                self.size = len(self.paths)
                
        # open mean / std from file
        with open(mean_std_cache, 'rb') as f:
            transform_dict = pickle.load(f)
            print('Loaded mean/std, # channels, # edges from cache.')
            self.mean = transform_dict['mean']
            self.std = transform_dict['std']
            self.ninput_channels = transform_dict['ninput_channels']
            self.ninput_edges = transform_dict['max_edges']
            
def collate_fn(batch):
    """Creates mini-batch tensors: list of values 
    We should build custom collate_fn rather than using default collate_fn
    """
    meta = {}
    
    # Edge case: entire batch is None 
    keys = None 
    for b in batch:
        if b: 
            keys = b.keys()
            break 
    if not keys:
        return None
    
    for key in keys:
        meta.update({key: [d[key] for d in batch if d is not None]})
    return meta
        