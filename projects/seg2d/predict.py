import os

import tensorflow as tf
from matplotlib import pyplot as plt

from loader import H5DataLoader
import utils

save_path = "results"
epoch = 79
seg_net_imported = tf.saved_model.load(os.path.join(save_path, 'epoch{:d}'.format(epoch)))


data_files = {'test': 'dataset/test.h5', 'val': 'dataset/val.h5'}
loader_val = H5DataLoader(data_files["val"], 3, training=False)
loader_test = H5DataLoader(data_files["test"], 3, training=False)


losses_all, metrics_all = [], []
for idx, (frames_val, masks_val) in enumerate(loader_test):
    
    if idx%5 ==0:
        preds_val = seg_net_imported(frames_val, training=False)

        losses = utils.dice_loss(preds_val, masks_val)
        dices, false_positives = utils.dice_metric_fg(preds_val, masks_val)
        print(losses)
        print(dices, false_positives)

        for dd in range(preds_val.shape[0]):
            plt.figure()
            plt.imshow(preds_val[dd,...])
            plt.figure()
            plt.imshow(masks_val[dd,...])
            plt.figure()
            plt.imshow(frames_val[dd,...])
        plt.show()

