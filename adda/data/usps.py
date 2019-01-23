import gzip
import os
from urllib.parse import urljoin

import numpy as np

from adda.data import DatasetGroup
from adda.data import ImageDataset
from adda.data import util
from adda.data.dataset import register_dataset


@register_dataset('usps')
class USPS(DatasetGroup):
    """USPS handwritten digits.

    Homepage: http://statweb.stanford.edu/~hastie/ElemStatLearn/data.html

    Images are 16x16 grayscale images in the range [0, 1].
    """

    base_url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'

    data_files = {
        'train': 'zip.train.gz',
        'test': 'zip.test.gz'
        }

    num_classes = 10

    def __init__(self, path=None, shuffle=True, download=True, selected=list(range(10)), ratios=None):
        DatasetGroup.__init__(self, 'usps', path=path, download=download)
        self.image_shape = (16, 16, 1)
        self.label_shape = ()
        self.shuffle = shuffle
        self.selected = selected
        self.num_classes = len(selected)
        self.label_dict = {}
        for i,l in enumerate(selected):
            self.label_dict[l] = i
        self.ratios = ratios
        self._load_datasets()

    def download(self):
        data_dir = self.get_path()
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for filename in self.data_files.values():
            path = self.get_path(filename)
            if not os.path.exists(path):
                url = urljoin(self.base_url, filename)
                util.maybe_download(url, path)

    def _load_datasets(self):
        abspaths = {name: self.get_path(path)
                    for name, path in self.data_files.items()}
        train_images, train_labels = self._read_datafile(abspaths['train'])
        test_images, test_labels = self._read_datafile(abspaths['test'])

        train_mask = [l in self.selected for l in train_labels]
        train_images = train_images[train_mask]
        train_labels = train_labels[train_mask]
        test_mask = [l in self.selected for l in test_labels]
        test_images = test_images[test_mask]
        test_labels = test_labels[test_mask]

        if self.ratios is not None:
            assert len(self.ratios) == self.num_classes
            # nmax = 540
            indices = []
            for l,r in zip(self.selected, self.ratios):
                inds = np.argwhere( train_labels==l )
                n_sel = int(len(inds)*r)
                indices.append( np.random.choice(inds.flatten(),n_sel,replace=False).flatten() )
            indices = np.concatenate(indices)
            train_images = train_images[indices]
            train_labels = train_labels[indices]

        train_labels = np.array([self.label_dict[l] for l in train_labels])
        test_labels = np.array([self.label_dict[l] for l in test_labels])

        self.train = ImageDataset(train_images, train_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)
        self.test = ImageDataset(test_images, test_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)

    def _read_datafile(self, path):
        """Read the proprietary USPS digits data file."""
        labels, images = [], []
        with gzip.GzipFile(path) as f:
            for line in f:
                vals = line.strip().split()
                labels.append(float(vals[0]))
                images.append([float(val) for val in vals[1:]])
        labels = np.array(labels, dtype=np.int32)
        labels[labels == 10] = 0  # fix weird 0 labels
        images = np.array(images, dtype=np.float32).reshape(-1, 16, 16, 1)
        images = (images + 1) / 2
        return images, labels


@register_dataset('usps1800')
class USPS1800(USPS):

    name = 'usps1800'

    def __init__(self, seed=None, path=None, shuffle=True):
        if seed is None:
            self.seed = hash(self.name) & 0xffffffff
        else:
            self.seed = seed
        USPS.__init__(self, path=path, shuffle=shuffle)

    def _load_datasets(self):
        abspaths = {name: self.get_path(path)
                    for name, path in self.data_files.items()}
        rand = np.random.RandomState(self.seed)
        train_images, train_labels = self._read_datafile(abspaths['train'])
        inds = rand.permutation(len(train_images))[:1800]
        inds.sort()
        train_images = train_images[inds]
        train_labels = train_labels[inds]
        test_images, test_labels = self._read_datafile(abspaths['test'])
        self.train = ImageDataset(train_images, train_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)
        self.test = ImageDataset(test_images, test_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)
