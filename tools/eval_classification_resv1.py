import logging
import os
import time
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import adda
from adda.data.office31 import office31
from adda.models.alexnet import AlexNet

# import seaborn as sn
# import pandas as pd
# import matplotlib.pyplot as plt
from nets.resnet_v1 import resnet_v1_50
from nets import resnet_v1
from tensorflow.python import pywrap_tensorflow

slim = tf.contrib.slim

def format_array(arr):
    return '    '.join(['{:.3f}'.format(x) for x in arr])


@click.command()
@click.option('--dataset', default='data/office31')
@click.option('--split', default='dslr')
@click.option('--model', default='resnet_v1_50')
@click.option('--weights', default='snapshot_/res1_office31_amazon')
@click.option('--gpu', default='0')
def main(dataset, split, model, weights, gpu):
    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    selected = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']
    ds = office31('data/office31', split ,selected=selected)
    split = getattr(ds, 'train')
    img, label = split.tf_ops()
    img = ds.preprocess(img, istraining=False)
    im_batch, label_batch = tf.train.batch([img, label], batch_size=1)

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, _ = resnet_v1_50(im_batch, 31, is_training=False, scope=model)
    net = tf.argmax(logits, -1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    var_dict = adda.util.collect_vars(model)
    restorer = tf.train.Saver(var_list=var_dict)
    if os.path.isdir(weights):
        weights = tf.train.latest_checkpoint(weights)
    logging.info('Evaluating {}'.format(weights))
    restorer.restore(sess, weights)
    #
    confusion = np.zeros((31,31), dtype=int)
    class_correct = np.zeros(31, dtype=np.int32)
    class_counts = np.zeros(31, dtype=np.int32)
    for i in tqdm(range(len(split))):
        predictions, gt = sess.run([net, label_batch])
        class_counts[gt[0]] += 1
        if predictions[0] == gt[0]:
            class_correct[gt[0]] += 1
        confusion[gt[0],predictions[0]] += 1
    logging.info('Class accuracies:')
    logging.info('    ' + '  '.join(ds.selected))
    logging.info('    ' + format_array(class_correct / class_counts))
    logging.info('Overall accuracy:')
    logging.info('  {}  '.format(np.sum(class_correct) / np.sum(class_counts)))
    print(format_array(class_counts))

    coord.request_stop()
    coord.join(threads)
    sess.close()
    

if __name__ == '__main__':
    main()
