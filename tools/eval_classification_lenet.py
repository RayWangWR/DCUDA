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


def format_array(arr):
    return '  '.join(['{:.3f}'.format(x) for x in arr])


@click.command()
@click.option('--dataset', default='mnist')
@click.option('--split', default='test')
@click.option('--model', default='lenet')
@click.option('--weights', default='snapshot/lenet_svhn')
@click.option('--gpu', default='0')
def main(dataset, split, model, weights, gpu):
    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    selected = list(range(10))
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
    

if __name__ == '__main__':
    main()
