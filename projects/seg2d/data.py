import os
import random
import shutil

import numpy as np
import nibabel as nib
import h5py


data_path = '../../../in4it_data/trus_processed_uint8/'
image_path = 'images'
label_path = ['seg_01','seg_02','seg_03']
save_path = 'dataset'
split_ratio = [8,1,1]  #[train,val,test]


if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
h5_id = {k:h5py.File(os.path.join(save_path,k+".h5"),'a') for k in ["train","val","test"]}

image_files = os.listdir(os.path.join(data_path,image_path))
split_ratio = [r/sum(split_ratio) for r in split_ratio]
num_files = [int(n*len(image_files)) for n in split_ratio[0:2]]
random.shuffle(image_files)

frame_idx = 0
for fidx,fn in enumerate(image_files):
    if fidx<num_files[0]:
        dataset = 'train'
    elif fidx<sum(num_files[0:2]):
        dataset = 'val'
    else:
        dataset = 'test'

    fobj = nib.load(os.path.join(data_path,image_path,fn))
    num_frames = fobj.shape[2]
    frames = np.asarray(fobj.dataobj) if (len(fobj.shape)==3) else np.asarray(fobj.dataobj)[...,np.newaxis]

    lobjs = [nib.load(os.path.join(data_path,lpath,'label_'+fn)) for lpath in label_path]
    labels = np.stack([np.asarray(obj.dataobj) if (len(obj.shape)==3) else np.asarray(obj.dataobj)[...,np.newaxis] for obj in lobjs],axis=3)
    if labels.shape[0:3] != fobj.shape:
        raise('shape inconsistent between labels and images')

    for dd in range(fobj.shape[2]):
        h5_id[dataset].create_dataset('/frame_%05d' % frame_idx, fobj.shape[0:2], dtype=frames.dtype, data=frames[:,:,dd])
        for ll in range(labels.shape[3]):
            h5_id[dataset].create_dataset('/label_%05d_%02d' % (frame_idx,ll), fobj.shape[0:2], dtype=labels.dtype, data=labels[:,:,dd,ll])
        frame_idx += 1

for fid in h5_id.values():
    fid.flush()
    fid.close()

print('%d frames saved at %s.' % (frame_idx,os.path.abspath(save_path)))
