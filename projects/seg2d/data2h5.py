import os

import numpy as np
import nibabel as nib
import h5py



resample_rate = 5

data_path = '../../../in4it-data/trus_processed_uint8/'
image_path = 'images'
label_path = ['seg_01','seg_02','seg_03']
filename_h5 = 'dataset%d' % (resample_rate*1e3) +".h5"
h5_id = h5py.File(filename_h5,'a')

image_files = os.listdir(os.path.join(data_path,image_path))
for fidx,fn in enumerate(image_files):

    fobj = nib.load(os.path.join(data_path,image_path,fn))
    frames = np.asarray(fobj.dataobj) if (len(fobj.shape)==3) else np.asarray(fobj.dataobj)[...,np.newaxis]

    lobjs = [nib.load(os.path.join(data_path,lpath,'label_'+fn)) for lpath in label_path]
    labels = np.stack([np.asarray(obj.dataobj) if (len(obj.shape)==3) else np.asarray(obj.dataobj)[...,np.newaxis] for obj in lobjs],axis=3)
    if labels.shape[0:3] != fobj.shape:
        raise('shape inconsistent between labels and images')
    
    # resample frames
    frames = frames[::resample_rate,::resample_rate,...]
    labels = labels[::resample_rate,::resample_rate,...]

    for dd in range(fobj.shape[2]):
        h5_id.create_dataset('/frame_%04d_%03d' % (fidx,dd), frames.shape[0:2], dtype=frames.dtype, data=frames[:,:,dd]) # [frame_case_frame]
        for ll in range(labels.shape[3]):
            h5_id.create_dataset('/label_%04d_%03d_%02d' % (fidx,dd,ll), labels.shape[0:2], dtype=labels.dtype, data=labels[:,:,dd,ll]) # [label_case_frame_obs]

h5_id.flush()
h5_id.close()

print('Frames saved as %s.' % filename_h5)
