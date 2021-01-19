
import os

import tensorflow as tf
import h5py

from loader import H5DataLoader 
from network import ResUNet
import utils


os.environ["CUDA_VISIBLE_DEVICES"]="0"

## settings
minibatch_size = 4
network_size = 16
learning_rate = 1e-3
num_epochs = 100
freq_info = 1
freq_val = 10
save_path = "results"

if not os.path.exists(save_path):
    os.makedirs(save_path)

## data loader
data_files = {
    'train': 'dataset/train.h5',
    'val': 'dataset/val.h5',
    'test': 'dataset/test.h5'
    }

loader_train = H5DataLoader(data_files["train"], minibatch_size, training=True)
loader_val = H5DataLoader(data_files["val"], 3, training=False)
# loader_test = H5DataLoader(data_files["test"], 1, training=False)


## network
seg_net = ResUNet(init_ch=network_size)
seg_net = seg_net.build(input_shape=loader_train.frame_size+(1,))


## train
optimizer = tf.optimizers.Adam(learning_rate)

# train step
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predicts = seg_net(images, training=True)
        loss = tf.reduce_mean(utils.dice_loss(predicts, labels))
    gradients = tape.gradient(loss, seg_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, seg_net.trainable_variables))
    return loss

# validation step
@tf.function
def val_step(images, labels):
    predicts = seg_net(images, training=False)
    losses = utils.dice_loss(predicts, labels)
    metrics = utils.dice_binary(predicts, labels)
    return losses, metrics, predicts

# train data batching
for epoch in range(num_epochs):
    for frames, masks in loader_train:
        loss_train = train_step(frames, masks)
    
    if (epoch+1) % freq_info == 0:
        tf.print('Epoch {}: loss={:0.5f}'.format(epoch,loss_train))

    if (epoch+1) % freq_val == 0:
        h5file = h5py.File(os.path.join(save_path,"epoch-{:05d}.h5".format(epoch)),'a')
        losses_val_all, metrics_all = [], []
        for idx, (frames_val, masks_val) in enumerate(loader_val):
            losses_val, metrics, preds_val = train_step(frames_val, masks_val)
            losses_val_all += [losses_val]
            metrics_all += [metrics]
            for dd in preds_val.shape[0]:
                h5file.create_dataset(
                    "/frame_%05d" % idx*preds_val.shape[0]+dd,
                    preds_val.shape[1:3],
                    dtype = preds_val.dtype,
                    data = preds_val[dd,...]
                    )    
        tf.print('Epoch {}: val-loss={:0.5f}, val-metric={:0.5f}'.format(
            tf.reduce_mean(tf.stack(losses_val_all,axis=0)),
            tf.reduce_mean(tf.stack(metrics_all,axis=0))
            ))
        h5file.flush()
        h5file.close()
