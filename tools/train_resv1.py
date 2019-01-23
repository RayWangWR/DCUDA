import logging
import os
import random
import sys
from collections import deque
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf
# from tensorflow.contrib import slim
from tqdm import tqdm

import adda
from adda.data.office31 import office31
from adda.models.alexnet import AlexNet
from nets.resnet_v1 import resnet_v1_50
from nets import resnet_v1
from tensorflow.python import pywrap_tensorflow

slim = tf.contrib.slim

def format_array(arr):
    return '    '.join(['{:.3f}'.format(x) for x in arr])

@click.command()
@click.argument('dataset')
@click.argument('split')
@click.argument('model')
@click.argument('output')
@click.option('--gpu', default='0')
@click.option('--iterations', default=1000)
@click.option('--batch_size', default=50)
@click.option('--display', default=10)
@click.option('--lr', default=1e-4)
@click.option('--stepsize', type=int)
@click.option('--snapshot', default=200)
@click.option('--weights')
@click.option('--weights_end')
@click.option('--ignore_label', type=int)
@click.option('--solver', default='sgd')
# @click.option('--seed', type=int)      #lr=5e-5
def main(dataset='data/office31', split='amazon', model='resnet_v1', output='res1_office31_amazon', gpu='1', iterations=20000, batch_size=64, display=10,
         lr=5e-5, stepsize=None, snapshot=200, weights=True, weights_end=None, ignore_label=None, solver='sgd'):
         # seed=None):
    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    ds = office31('data/office31',split)#,selected=['bike_helmet'])
    split =  getattr(ds,'train')
    img, label = split.tf_ops()
    img = ds.preprocess(img, istraining=True)
    im_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size)

    ds_eval = office31('data/office31', 'dslr')  # ,selected=['bike_helmet'])
    split_eval = getattr(ds_eval, 'train')
    img_eval, label_eval = split_eval.tf_ops()
    img_eval = ds_eval.preprocess(img_eval, istraining=False)
    im_batch_eval, label_batch_eval = tf.train.batch([img_eval, label_eval], batch_size=1)

    im_batch_h = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    label_batch_h = tf.placeholder(tf.int32, [batch_size,])
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, _ = resnet_v1_50(im_batch_h, 31, is_training=True, scope=model)
    model_vars = adda.util.collect_vars(model)

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits_eval, _ = resnet_v1_50(im_batch_eval, 31, is_training=False, scope=model, reuse=True)
    predictions = tf.argmax(logits_eval, -1)

    class_loss = tf.losses.sparse_softmax_cross_entropy(label_batch_h, logits=logits)
    loss = tf.losses.get_total_loss()

    lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
    if solver == 'sgd':
        optimizer = tf.train.MomentumOptimizer(lr_var, 0.99)
    else:
        optimizer = tf.train.AdamOptimizer(lr_var)
    step = optimizer.minimize(loss)#, var_list=vars_train)

    config = tf.ConfigProto(device_count=dict(GPU=2))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())

    # Load weights
    skip_vars = ['logits/weights', 'logits/biases']
    ns = list(model_vars.keys())
    path = './resnet_v1_slim/resnet_v1_50.ckpt'
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    for n in ns:
        if n in skip_vars:
            continue
        print('Restoring: {}'.format(n))
        model_var = model_vars[n]
        name_in_c = 'resnet_v1_50/' + n
        try:
            check_var = reader.get_tensor(name_in_c)
        except:
            print('{} not in the checkpoint'.format(name_in_c))
            continue
            # check_var = np.zeros(model_var.get_shape())
        sess.run(model_var.assign(check_var))


    # model_vars = adda.util.collect_vars(model)
    saver = tf.train.Saver(var_list=model_vars, max_to_keep=1000)
    output_dir = os.path.join('snapshot', output)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    losses = deque(maxlen=10)
    bar = tqdm(range(iterations))
    bar.set_description('{} (lr: {:.0e})'.format(output, lr))
    bar.refresh()
    for i in bar:
        # loss_val, _ = sess.run([loss, step])
        im_batch_v, label_batch_v = sess.run([im_batch,label_batch])
        loss_val, logits_val, _ = sess.run([loss, logits, step], feed_dict={im_batch_h: im_batch_v, label_batch_h: label_batch_v})
        preds = np.argmax(logits_val, axis=1).flatten()
        print(preds)
        print('Accuracy: {}'.format(np.mean(preds==label_batch_v.flatten())))
        losses.append(loss_val)
        if i % display == 0:
            logging.info('{:20} {:10.4f}     (avg: {:10.4f})'
                        .format('Iteration {}:'.format(i),
                                loss_val,
                                np.mean(losses)))
        if stepsize is not None and (i + 1) % stepsize == 0:
            lr = sess.run(lr_var.assign(lr * 0.1))
            logging.info('Changed learning rate to {:.0e}'.format(lr))
            bar.set_description('{} (lr: {:.0e})'.format(output, lr))
        if (i + 1) % snapshot == 0:
            snapshot_path = saver.save(sess, os.path.join(output_dir, output),
                                       global_step=i + 1)
            logging.info('Saved snapshot to {}'.format(snapshot_path))
        if (i + 1) % 100 == 0:
            class_correct = np.zeros(31, dtype=np.int32)
            class_counts = np.zeros(31, dtype=np.int32)
            for i in tqdm(range(len(split_eval))):
                predictions_val, gt = sess.run([predictions, label_batch_eval])
                class_counts[gt[0]] += 1
                if predictions_val[0] == gt[0]:
                    class_correct[gt[0]] += 1
            logging.info('Class accuracies:')
            logging.info('    ' + '  '.join(ds_eval.selected))
            logging.info('    ' + format_array(class_correct / class_counts))
            logging.info('Overall accuracy:')
            logging.info('    ' + str(np.sum(class_correct) / np.sum(class_counts)))
            print(format_array(class_counts))

    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    main()
