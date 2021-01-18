
from loader import H5DataLoader 

# data loader
data_files = {
    'train': 'dataset/train.h5',
    'val': 'dataset/val.h5',
    'test': 'dataset/test.h5'
    }

loader_train = H5DataLoader(data_files["train"], 16, training=True)
loader_val = H5DataLoader(data_files["val"], 3, training=False)
loader_test = H5DataLoader(data_files["test"], 1, training=False)



for frames, labels in loader_train:
    frames, labels
