import os
from urllib.parse import urljoin

import numpy as np
from scipy.io import loadmat

from adda.data import DatasetGroup
from adda.data import ImageDataset
from adda.data import util
from adda.data.dataset import register_dataset

import random
import matplotlib.pyplot as plt
import tensorflow as tf

@register_dataset('Fresno')
class Fresno(DatasetGroup):
    '''Fresno_training: chunks 1-10 (100000 panel:non_panel=22071:77979)
       Fresno_testing: chunk 12 (20000 panel:non_panel=5648:14352)'''


    data_files={'train':'Fresno_training.mat','test':'Fresno_testing.mat'}
    num_classes=2

    def __init__(self, path='/home/ray/adda-master/data/solar_panel',
                 shuffle=True):
        DatasetGroup.__init__(self, 'Fresno', path=path)
        self.train_on_extra = False
        # self.image_shape = (32, 32, 3)
        # self.label_shape = ()
        self.shuffle = shuffle
        self._load_datasets()

    def _load_datasets(self):
        abspaths = {name: self.get_path(path)
                    for name, path in self.data_files.items()}
        train_mat = loadmat(abspaths['train'])
        train_images = train_mat['patches'].transpose((3, 0, 1, 2))
        train_labels = train_mat['labels'].squeeze()
        if self.train_on_extra:
            extra_mat = loadmat(abspaths['extra'])
            train_images = np.vstack((train_images,
                                      extra_mat['patches'].transpose((3, 0, 1, 2))))
            train_labels = np.concatenate((train_labels,
                                           extra_mat['labels'].squeeze()))
        # train_labels[train_labels == 10] = 0
        train_images = train_images.astype(np.float32) / 255
        test_mat = loadmat(abspaths['test'])
        test_images = test_mat['patches'].transpose((3, 0, 1, 2))
        test_images = test_images.astype(np.float32) / 255
        test_labels = test_mat['labels'].squeeze()
        # print('train: {}. 0: {}, 1: {}'.format(len(train_labels), sum(train_labels == 0), sum(train_labels == 1)))
        # print('test: {}. 0: {}, 1: {}'.format(len(test_labels), sum(test_labels == 0), sum(test_labels == 1)))

        self.image_shape=train_images.shape[1:4]
        self.label_shape=()

        self.mean=train_mat['m'].astype(np.float32)/255
        train_images = train_images - self.mean
        test_images = test_images - self.mean

        self.train = ImageDataset(train_images, train_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)
        # self.train = ImageDataset(train_images, train_labels,
        #                           image_shape=self.image_shape,
        #                           label_shape=self.label_shape,
        #                           shuffle=self.shuffle)
        # labeled_indices = [i for i in range(0, len(train_images), round(1 / 0.002))]
        # self.train_labeled = ImageDataset(train_images[labeled_indices], train_labels[labeled_indices],
        #                                   image_shape=self.image_shape,
        #                                   label_shape=self.label_shape,
        #                                   shuffle=self.shuffle)
        self.test = ImageDataset(test_images, test_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)

# ds_obj = Fresno()
# ds = ds_obj.train
# im, label = ds.tf_ops()
# m = ds_obj.mean
# im_batch, label_batch = tf.train.batch([im, label], batch_size=10)
# config = tf.ConfigProto(device_count=dict(GPU=1))
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# sess.run(tf.global_variables_initializer())
# ims, lbs = sess.run([im_batch, label_batch])
# i = 0
# img = ((ims[i,:,:,:] + m)*255).astype(np.uint8)
# lb = lbs[i]
# plt.imshow(img)
# plt.title(lb)
# plt.show()


