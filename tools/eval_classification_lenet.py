import logging
import os
import time
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import adda
from adda.models.lenet import lenet

# import seaborn as sn
import pandas as pd
# import matplotlib.pyplot as plt

def format_array(arr):
    return '  '.join(['{:.3f}'.format(x) for x in arr])


@click.command()
# @click.argument('dataset')
# @click.argument('split')
# @click.argument('model')
# @click.argument('weights')
@click.option('--dataset', default='mnist')
@click.option('--split', default='test')
@click.option('--model', default='lenet')
@click.option('--weights', default='snapshot/lenet_svhn')
@click.option('--gpu', default='1')
def main(dataset, split, model, weights, gpu):
    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    selected = list(range(10))
    # ratios = None#[1,1]
    # svhn_counts = np.array([4948, 13861, 10585, 8497, 7458, 6882, 5727, 5595, 5045, 4659])
    # ratios = min(svhn_counts) / svhn_counts
    # mnist_counts = np.array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])
    # ratios = min(mnist_counts) / mnist_counts
    # ratios = ratios * np.linspace(1, 0.3, len(ratios))
    # usps_counts = np.array([1194, 1005, 731, 658, 652, 556, 664, 645, 542, 644])
    # ratios = min(usps_counts) / usps_counts
    # ratios = ratios * np.linspace(1, 0.3, len(ratios))
    dataset_name = dataset
    split_name = split
    dataset = adda.data.get_dataset(dataset, shuffle=False, selected=selected, ratios=None)
    split = getattr(dataset, split)
    model_fn = adda.models.get_model_fn(model)
    im, label = split.tf_ops(capacity=2)
    im = adda.models.preprocessing(im, model_fn)
    im_batch, label_batch = tf.train.batch([im, label], batch_size=1)

    net, layers = lenet(im_batch, n_classes=len(selected))#model_fn(im_batch, is_training=False)
    net = tf.argmax(net, -1)

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
    confusion = np.zeros((len(selected),len(selected)), dtype=int)
    class_correct = np.zeros(dataset.num_classes, dtype=np.int32)
    class_counts = np.zeros(dataset.num_classes, dtype=np.int32)
    for i in tqdm(range(len(split))):
        predictions, gt = sess.run([net, label_batch])
        class_counts[gt[0]] += 1
        if predictions[0] == gt[0]:
            class_correct[gt[0]] += 1
        confusion[gt[0],predictions[0]] += 1
    logging.info('Class accuracies:')
    logging.info('    ' + format_array(class_correct / class_counts))
    logging.info('Overall accuracy:')
    logging.info('    ' + str(np.sum(class_correct) / np.sum(class_counts)))
    print(format_array(class_counts))

    coord.request_stop()
    coord.join(threads)
    sess.close()

    # selected = [2,8]
    # n_corrects = 0
    # n_samples = 0
    # for l in selected:
    #     nc = confusion[l,l]
    #     na = sum(confusion[l])
    #     print('{}: {}'.format(l, nc/na))
    #     n_corrects += nc
    #     n_samples += na
    # print('Selected: {}'.format(n_corrects/n_samples))

    # confusion_df = pd.DataFrame(confusion, index=[str(l) for l in selected],
    #                             columns=[str(l) for l in selected])
    # plt.figure(figsize=(10, 10))
    # sn.heatmap(confusion_df, annot=True, cbar=True, fmt='d', cmap='OrRd')
    # plt.show()
    

if __name__ == '__main__':
    main()
