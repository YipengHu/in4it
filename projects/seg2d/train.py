
from loader import H5DataLoader 
from network import ResUNet



# data loader
data_files = {
    'train': 'dataset/train.h5',
    'val': 'dataset/val.h5',
    'test': 'dataset/test.h5'
    }

loader_train = H5DataLoader(data_files["train"], 16, training=True)
loader_val = H5DataLoader(data_files["val"], 3, training=False)
loader_test = H5DataLoader(data_files["test"], 1, training=False)

# network
image_size=(128,128)
seg_net = ResUNet(init_ch=32, num_levels=3, out_ch=1)  # output ddfs in x,y two channels
seg_net = seg_net.build(input_shape=image_size+(1,))
seg_net = seg_net.build(input_shape=loader_train.frame_size+(1,))


for frames, labels in loader_train:
    frames, labels
