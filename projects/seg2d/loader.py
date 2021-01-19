

import h5py
import random

import numpy as np

class H5DataLoader():
    def __init__(self, filename, batch_size, training=True):
        self.h5_file = h5py.File(filename,'r')  
        self.frame_keys = sorted([k for k in self.h5_file.keys() if k.split('_')[0]=='frame'])
        self.label_keys = sorted([k for k in self.h5_file.keys() if k.split('_')[0]=='label'])
        if len(self.label_keys) % len(self.frame_keys) != 0:  #divisible
            raise Exception('unequal numbers of frames and labels.')
        else:
            self.num_labels_per_frame = int(len(self.label_keys)/len(self.frame_keys))
        if not set([k.split('_')[1] for k in self.frame_keys]).issubset(set([k.split('_')[1] for k in self.label_keys])):
            raise Exception('not all frames are labeled.')
        self.frame_size = self.h5_file[self.frame_keys[0]].shape[0:2]
        
        self.training = training
        if not self.training:
            self.frame_keys = ['frame_'+k.split('_')[1] for k in self.label_keys]
        self.num_frames = len(self.frame_keys)

        self.batch_size = batch_size
        if (self.num_frames % self.batch_size != 0) & (not self.training):
            raise Warning('not all frame-label pairs will be tested.')
        self.num_batches = int(self.num_frames/self.batch_size) # skip the remainders
    
    def __iter__(self):
        self.batch_idx = 0
        if self.training:
            random.shuffle(self.frame_keys)
            self.label_keys = ['label_'+k.split('_')[1]+'_{:02d}'.format(random.randint(0,self.num_labels_per_frame-1)) for k in self.frame_keys]
        return self
    
    def __next__(self):
        if self.batch_idx>=self.num_batches:
            raise StopIteration
        idx0, idx1 = self.batch_idx*self.batch_size, (self.batch_idx+1)*self.batch_size
        frames = np.stack([self.h5_file[k][()].astype('float32') for k in self.frame_keys[idx0:idx1]], axis=0)
        labels = np.stack([self.h5_file[k][()].astype('float32') for k in self.label_keys[idx0:idx1]], axis=0)
        frames /= frames.max(axis=(1,2),keepdims=True)  #normalisation for unsigned data type
        # labels /= labels.max(axis=(1,2),keepdims=True) + np.finfo('float32').eps
        self.batch_idx += 1
        return frames, labels
