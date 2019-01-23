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

@register_dataset('amazon_reviews')
class amazon_reviews(DatasetGroup):

    num_classes=2

    def __init__(self, split, path='/home/ray/adda-master/data/amazon_reviews', num_f=5000, shuffle=True, pos_rate=1, neg_rate=1):
        self.num_f = 5000
        self.split = split
        self.path = path
        self.shuffle = shuffle
        self.pos_rate = pos_rate
        self.neg_rate = neg_rate
        self._load_datasets()

    def _load_datasets(self):
        data_train = np.load(os.path.join(self.path,self.split,'data_{}_{}.npy'.format('train',self.num_f))).item()
        train_fts = data_train['fts']
        train_labels = data_train['labels']

        inds_pos = np.argwhere(train_labels==1).flatten()
        inds_neg = np.argwhere(train_labels==0).flatten()
        inds_pos = [inds_pos[i] for i in range(0,len(inds_pos),self.pos_rate)]
        inds_neg = [inds_neg[i] for i in range(0,len(inds_neg),self.neg_rate)]
        inds = inds_pos + inds_neg
        train_fts = train_fts[inds]
        train_labels = train_labels[inds]


        data_test = np.load(os.path.join(self.path,self.split,'data_{}_{}.npy'.format('test',self.num_f))).item()
        test_fts = data_test['fts']
        test_labels = data_test['labels']

        self.image_shape=train_fts.shape[1:]
        self.label_shape=()

        self.train = ImageDataset(train_fts, train_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)

        self.test = ImageDataset(test_fts, test_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)

