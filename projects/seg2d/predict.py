import os
# import time

import tensorflow as tf
from matplotlib import pyplot as plt

from loader import H5DataLoader
import utils


save_path = "trained/any"
epoch = 249
seg_net_imported = tf.saved_model.load(os.path.join(save_path, 'epoch{:d}'.format(epoch)))

batch_size = 3
data_files = {'test': 'dataset/test.h5', 'val': 'dataset/val.h5'}
# loader_test = H5DataLoader(data_files["val"], batch_size, training=False)
loader_test = H5DataLoader(data_files["test"], batch_size, training=False)


# stats
losses_all, dices_all, false_positives_all = [], [], []
for idx, (frames_test, masks_test) in enumerate(loader_test):

    # t0 = time.time()
    preds_test = seg_net_imported(frames_test, training=False)
    # print('Inference time [%f]s' % (time.time()-t0))

    losses = utils.dice_loss(preds_test, masks_test)
    dices, false_positives = utils.dice_metric_fg(preds_test, masks_test)
    losses_all += [losses]
    dices_all += [dices]
    false_positives_all += [false_positives]

print('val-loss={:0.5f}, val-dice={:0.5f}, false_positives={:0.5f}'.format(
    tf.reduce_mean(tf.concat(losses_all,axis=0)),
    tf.reduce_mean(tf.concat(dices_all,axis=0)),
    tf.reduce_mean(tf.concat(false_positives_all,axis=0))
    ))


# visualise
for idx, (frames_test, masks_test) in enumerate(loader_test):
    
    if idx%5 ==0:
        preds_test = seg_net_imported(frames_test, training=False)

        losses = utils.dice_loss(preds_test, masks_test)
        dices, false_positives = utils.dice_metric_fg(preds_test, masks_test)
        print('losses: ',losses)
        print('dices: ', dices)
        print('false-spotives: ', false_positives)

        for dd in range(preds_test.shape[0]):
            plt.figure()
            plt.imshow(tf.concat([
                preds_test[dd,...],
                frames_test[dd,...],
                masks_test[dd,...]], axis=1))
        plt.show()
