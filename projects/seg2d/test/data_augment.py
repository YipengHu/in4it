
import os

from matplotlib import pyplot as plt
import numpy as np
from loader import H5DataLoader 
import utils


minibatch_size = 16
loader_train = H5DataLoader('dataset/train.h5', minibatch_size, training=True)

def montage_images(images):
    for n in range(int(images.shape[0]**(0.5))):
            if images.shape[0] % (n+1) == 0:
                nn = n+1
    images = np.reshape(images,(-1,images.shape[1]*nn,images.shape[2]))
    images = np.reshape(np.transpose(images,[0,2,1]),(-1,images.shape[1]))
    return images

for images, labels in loader_train: 
    images_warped, labels_warped = utils.random_image_label_transform(images, labels)

    plt.figure()
    plt.imshow(montage_images(images))
    plt.figure()
    plt.imshow(montage_images(images_warped))
    plt.figure()
    plt.imshow(montage_images(labels))
    plt.figure()
    plt.imshow(montage_images(labels_warped))
    plt.show()
