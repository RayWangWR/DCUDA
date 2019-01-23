import gzip
import operator
import os
import struct
from functools import reduce
from urllib.parse import urljoin

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import imageio

from adda.data import DatasetGroup
from adda.data import ImageDataset
from adda.data import util
from adda.data.dataset import register_dataset

import tensorflow as tf

class visda2017:
    def __init__(self, path, split, shuffle=True, img_shape=(224,224,3), selected=None):
        self.path = path
        self.split = split
        self.shuffle = shuffle
        if selected is None:
            self.selected = list(range(12))
        else:
            self.selected = selected
        self.num_classes = len(self.selected)
        self.image_shape = img_shape#(224,224,3)
        self.label_shape = ()
        if split == 'train':
            self.m = np.array([225.17808756, 225.83161464, 226.16370891])
        elif split == 'validation':
            self.m = np.array([97.49751596, 102.99828336, 107.61180433])
        else:
            raise Exception('split should be "train" or "validation".')
        self._load_datasets()

    def _load_datasets(self):
        with open(os.path.join(self.path,self.split,'image_list.txt'), 'r') as f:
            lines = [l[:-1] for l in f.readlines()]
        imgs = []
        labels = []
        for l in lines:
            img_n, label = l.split(' ')
            label = eval(label)
            if label not in self.selected:
                continue
            # img = cv2.imread(os.path.join(self.path,self.split,img_n)).astype(np.float32)
            # img = cv2.resize(img, self.image_shape[:2])
            # img = img - self.m
            imgs.append(os.path.join(self.path, self.split, img_n))
            labels.append(label)
        imgs = np.array(imgs)#np.stack(imgs, axis=0)
        labels = np.array(labels)


        self.train = ImageDataset(imgs, labels,
                                  image_shape=(),#self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)

    def preprocess(self, img_n):
        img_string = tf.read_file(img_n)
        if self.split == 'train':
            img = tf.image.decode_png(img_string)
        else:
            img = tf.image.decode_jpeg(img_string)
            # img = tf.image.decode_image(img_string, channels=3)
        img = tf.image.resize_images(img, self.image_shape[:2])
        img = img - self.m
        return img




# # for testing
# ds = visda2017('data/visda2017','validation')#,selected=['bike_helmet'])
# # split =  getattr(ds,'train')
# # img, label = split.tf_ops(img_type=tf.string)
# # img = ds.preprocess(img)
# # im_batch, label_batch = tf.train.batch([img, label], batch_size=10)
#
# ds_source = visda2017('data/visda2017', 'train')
# source_split = getattr(ds_source, 'train')
# source_im, source_label = source_split.tf_ops(img_type=tf.string)
# source_im = ds_source.preprocess(source_im)
# source_im_batch, source_label_batch = tf.train.batch([source_im, source_label], batch_size=10)
#
# ds_target = visda2017('data/visda2017', 'validation')
# target_split = getattr(ds_target, 'train')
# target_im, target_label = target_split.tf_ops(img_type=tf.string)
# target_im = ds_target.preprocess(target_im)
# target_im_batch, target_label_batch = tf.train.batch([target_im, target_label], batch_size=10)
#
#
# config = tf.ConfigProto(device_count=dict(GPU=1))
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# # img_n, l = sess.run([img, label])
# # print(img_n, l)
# source_ims, target_ims = sess.run([source_im_batch,target_im_batch])
# imgs, labels = sess.run([target_im_batch,target_label_batch])
# im = imgs[0]
# lb = labels[0]
# im = im + ds.m
# print(lb)
# plt.imshow(im.astype(np.uint8))
# plt.show()