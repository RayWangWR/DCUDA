import gzip
import operator
import os
import struct
from functools import reduce
from urllib.parse import urljoin

import numpy as np
import cv2
# import matplotlib.pyplot as plt
import tensorflow as tf
import imageio

from adda.data import DatasetGroup
from adda.data import ImageDataset
from adda.data import util
from adda.data.dataset import register_dataset

class office31:
    def __init__(self, path, split, shuffle=True, img_shape=(224,224,3), selected=None):
        self.path = path
        self.split = split
        self.shuffle = True
        self.all_classes = np.array(os.listdir( os.path.join(path,split,'images') ))
        if selected is None:
            self.selected = self.all_classes
            self.selected_labels = np.array(list(range(len(self.selected))))
        else:
            self.selected = selected
            self.selected_labels = []
            for s in self.selected:
                assert s in self.all_classes
                self.selected_labels.append( np.argwhere(self.all_classes==s) )
            self.selected_labels = np.array(self.selected_labels).flatten()
        self.num_classes = len(self.selected)
        self.image_shape = img_shape#(224,224,3)
        self.label_shape = ()
        self._load_datasets()

    def _load_datasets(self):
        imgs = []
        labels = []
        for i,c in zip(self.selected_labels,self.selected):#enumerate(self.selected):
            for img_name in os.listdir( os.path.join(self.path,self.split,'images',c) ):
                img = np.array(imageio.imread(os.path.join( self.path,self.split,'images',c,img_name ))).astype(np.float32)
                img = cv2.resize(img.astype(np.float32), (256,256))
                # margin = int((256-227)/2)
                # img = img[margin:margin+227,margin:margin+227,:]
                # img = img - np.array([0.485, 0.456, 0.406])*255
                # img[:,:,0], img[:,:,2] = img[:,:,2], img[:,:,0]
                if self.split == 'dslr':
                    img = (img - np.array([120.07020481, 114.4072377, 103.62625268]))# / np.array([47.27234164, 45.61725823, 46.38534723])
                elif self.split == 'webcam':
                    img = (img - np.array([156.05485446, 157.78500424, 157.41063227]))# / np.array([58.0478348, 59.51542406, 60.49295001])
                elif self.split == 'amazon':
                    img = (img - np.array([202.04944002, 200.48261293, 199.9658089 ]))# / np.array([70.61369137, 71.78848969, 72.13260695])
                else:
                    raise Exception('Wrong split name')
                img = img[:,:,[2,1,0]]      # Originally exists
                imgs.append(img)
                labels.append(i)
        imgs = np.stack(imgs,axis=0)
        labels = np.stack(labels,axis=0)

        self.train = ImageDataset(imgs, labels,
                                  image_shape=(256,256,3),
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)

    def preprocess(self, img, istraining=True):
        assert img.shape[2] == self.image_shape[2]
        if istraining:
            top = np.random.choice(img.shape[0]-self.image_shape[0])
            left = np.random.choice(img.shape[1] - self.image_shape[1])
        else:
            top = int( (int(img.shape[0])-self.image_shape[0])/2 )
            left = int( (int(img.shape[1])-self.image_shape[1])/2 )
        img = tf.slice(img, [top, left, 0], self.image_shape)
        if istraining:
            img = tf.image.random_flip_left_right(img)
        return img


# # for testing
# ds = office31('data/office31','dslr')#,selected=['bike_helmet'])
# split =  getattr(ds,'train')
# img, label = split.tf_ops()
# img = ds.preprocess(img, istraining=False)
# im_batch, label_batch = tf.train.batch([img, label], batch_size=10)
# config = tf.ConfigProto(device_count=dict(GPU=1))
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# imgs, labels = sess.run([im_batch,label_batch])
# im = imgs[0]
# lb = labels[0]
# # im[:,:,0], im[:,:,2] = im[:,:,2], im[:,:,0]
# im = im[:,:,[2,1,0]]
# im = im + np.array([0.485, 0.456, 0.406])*255
# print(ds.all_classes[lb])
# plt.imshow(im.astype(np.uint8))
# plt.show()