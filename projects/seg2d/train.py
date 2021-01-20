import os

import tensorflow as tf

from loader import H5DataLoader
from network import ResUNet
import utils


os.environ["CUDA_VISIBLE_DEVICES"]="0"

## settings
minibatch_size = 32
network_size = 16
learning_rate = 1e-4
num_epochs = 200
freq_info = 1
freq_save = 20
save_path = "results"

if not os.path.exists(save_path):
    os.makedirs(save_path)

## data loader
data_files = {'train': 'dataset/train.h5', 'val': 'dataset/val.h5'}
loader_train = H5DataLoader(data_files["train"], minibatch_size, training=True)
loader_val = H5DataLoader(data_files["val"], 3, training=False)


## network
seg_net = ResUNet(init_ch=network_size)
seg_net = seg_net.build(input_shape=loader_train.frame_size+(1,))


## train
optimizer = tf.optimizers.Adam(learning_rate)

@tf.function
def train_step(images, labels):  # train step
    with tf.GradientTape() as tape:
        # images, labels = tf.convert_to_tensor(images), tf.convert_to_tensor(labels)
        images, labels = utils.random_image_label_transform(images, labels)
        predicts = seg_net(images, training=True)
        loss = tf.reduce_mean(utils.dice_loss(predicts, labels))
    gradients = tape.gradient(loss, seg_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, seg_net.trainable_variables))
    return loss

@tf.function
def val_step(images, labels):  # validation step
    predicts = seg_net(images, training=False)
    losses = utils.dice_loss(predicts, labels)
    metrics = utils.dice_binary(predicts, labels)
    return losses, metrics

for epoch in range(num_epochs):
    for frames, masks in loader_train: 
        loss_train = train_step(frames, masks)

    if (epoch+1) % freq_info == 0:
        tf.print('Epoch {}: loss={:0.5f}'.format(epoch,loss_train))

    if (epoch+1) % freq_save == 0:
        losses_val_all, metrics_all = [], []
        for idx, (frames_val, masks_val) in enumerate(loader_val):
            losses_val, metrics = val_step(frames_val, masks_val)
            losses_val_all += [losses_val]
            metrics_all += [metrics]
        tf.print('Epoch {}: val-loss={:0.5f}, val-metric={:0.5f}'.format(
            epoch,
            tf.reduce_mean(tf.concat(losses_val_all,axis=0)),
            tf.reduce_mean(tf.concat(metrics_all,axis=0))
            ))
        tf.saved_model.save(seg_net, os.path.join(save_path, 'epoch{:d}'.format(epoch)))
        tf.print('Model saved.')
