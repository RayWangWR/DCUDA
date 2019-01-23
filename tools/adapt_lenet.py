import logging
import os
import random
from collections import deque
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tqdm import tqdm

import adda

import matplotlib.pyplot as plt


@click.command()
@click.option('--source', default='svhn:train')
@click.option('--target', default='mnist:train')
@click.option('--model', default='lenet')
@click.option('--output', default='adapt_lenet_svhn_mnist')
@click.option('--gpu', default='0')
@click.option('--iterations', default=900)
@click.option('--batch_size', default=128)
@click.option('--display', default=10)
@click.option('--lr', default= 0.0002)
@click.option('--stepsize', type=int)
@click.option('--snapshot', default=100)
@click.option('--weights', required=True, default='snapshot/lenet_svhn')
@click.option('--solver', default='adam')
@click.option('--adversary', 'adversary_layers', default=[500, 500],
              multiple=True)
@click.option('--adversary_leaky/--adversary_relu', default=True)
@click.option('--seed', type=int)
def main(source, target, model, output,
         gpu, iterations, batch_size, display, lr, stepsize, snapshot, weights,
         solver, adversary_layers, adversary_leaky, seed):

    centroids_path = os.path.join('snapshot',output,'means_10.npy')
    n_clusters = 10

    # miscellaneous setup
    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    if seed is None:
        seed = random.randrange(2 ** 32 - 2)
    logging.info('Using random seed {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed + 1)
    tf.set_random_seed(seed + 2)
    error = False
    try:
        source_dataset_name, source_split_name = source.split(':')
    except ValueError:
        error = True
        logging.error(
            'Unexpected source dataset {} (should be in format dataset:split)'
            .format(source))
    try:
        target_dataset_name, target_split_name = target.split(':')
    except ValueError:
        error = True
        logging.error(
            'Unexpected target dataset {} (should be in format dataset:split)'
            .format(target))
    if error:
        raise click.Abort

    selected = list(range(10))
    ratios = None#np.linspace(1,0.3,len(selected))

    # setup data
    logging.info('Adapting {} -> {}'.format(source, target))
    source_dataset = getattr(adda.data.get_dataset(source_dataset_name),
                             source_split_name)
    target_dataset = getattr(adda.data.get_dataset(target_dataset_name),#, selected=selected, ratios=ratios),
                             target_split_name)
    source_im, source_label = source_dataset.tf_ops()
    target_im, target_label = target_dataset.tf_ops()
    model_fn = adda.models.get_model_fn(model)
    source_im = adda.models.preprocessing(source_im, model_fn)
    target_im = adda.models.preprocessing(target_im, model_fn)
    source_im_batch, source_label_batch = tf.train.batch(
        [source_im, source_label], batch_size=batch_size)
    target_im_batch, target_label_batch = tf.train.batch(
        [target_im, target_label], batch_size=batch_size)

    val_dataset = getattr(adda.data.get_dataset(target_dataset_name),#, selected=selected),
                             'test')
    val_im, val_label = val_dataset.tf_ops()
    val_im = adda.models.preprocessing(val_im, model_fn)
    val_im_batch, val_label_batch = tf.train.batch(
        [val_im, val_label], batch_size=1)

    # base network
    source_im_batch_h = tf.placeholder(tf.float32, shape=(batch_size,28,28,1))
    target_im_batch_h = tf.placeholder(tf.float32, shape=(batch_size,28,28,1))
    source_ft, _ = model_fn(source_im_batch_h, scope='source')
    target_ft, _ = model_fn(target_im_batch_h, scope='target')
    val_ft, _ = model_fn(val_im_batch, is_training=False, scope='val')
    val_out = tf.argmax(val_ft, -1)

    # adversarial network
    source_ft = tf.reshape(source_ft, [-1, int(source_ft.get_shape()[-1])])
    target_ft = tf.reshape(target_ft, [-1, int(target_ft.get_shape()[-1])])
    adversary_ft = tf.concat([source_ft, target_ft], 0)
    source_adversary_label = tf.zeros([tf.shape(source_ft)[0]], tf.int32)
    target_adversary_label = tf.ones([tf.shape(target_ft)[0]], tf.int32)
    adversary_label = tf.concat(
        [source_adversary_label, target_adversary_label], 0)
    adversary_logits = adda.adversary.adversarial_discriminator(
        adversary_ft, adversary_layers, leaky=adversary_leaky)

    # adda losses
    weights_ins = tf.placeholder(tf.float32, shape=(2*batch_size))
    mapping_loss = tf.losses.sparse_softmax_cross_entropy(
        1 - adversary_label, adversary_logits)
    adversary_loss = tf.losses.sparse_softmax_cross_entropy(
        adversary_label, adversary_logits, weights=weights_ins)

    # kl_loss
    centers = tf.Variable(np.zeros((n_clusters,10),dtype=np.float32), dtype=tf.float32, trainable=True)
    assert centers.get_shape()[0] == n_clusters
    assert centers.get_shape()[1] == target_ft.shape[1]
    Q = []
    for i in range(batch_size):
        row = []
        for j in range(centers.get_shape()[0]):
            ft_i = tf.gather(target_ft, i)
            c_j = tf.gather(centers, j)
            q_ij = 1 / (1 + tf.square(tf.norm(ft_i - c_j)))
            row.append(q_ij)
        row = tf.stack(row, axis=0)
        row = row / tf.reduce_sum(row)
        Q.append(row)
    Q = tf.stack(Q, axis=0)
    P = tf.placeholder(dtype=tf.float32, shape=(batch_size, n_clusters))
    # KL divergence
    I = P * tf.log(P / Q)
    kl_loss = tf.reduce_sum(I)

    # dissimiliar loss
    centers_prob = tf.contrib.layers.softmax(centers)
    sim_mat = tf.matmul(centers_prob, centers_prob, transpose_b=True)
    diss_loss = tf.sqrt(tf.reduce_sum(tf.square(sim_mat -tf.diag( tf.diag_part(sim_mat) ))))


    # variable collection
    source_vars = adda.util.collect_vars('source')
    target_vars = adda.util.collect_vars('target')
    adversary_vars = adda.util.collect_vars('adversary')

    val_vars = adda.util.collect_vars('val')

    # optimizer
    lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
    if solver == 'sgd':
        optimizer = tf.train.MomentumOptimizer(lr_var, 0.99)
    else:
        optimizer = tf.train.AdamOptimizer(lr_var, 0.5)
    mapping_step = optimizer.minimize(
        mapping_loss, var_list=list(target_vars.values()))
    adversary_step = optimizer.minimize(
        adversary_loss, var_list=list(adversary_vars.values()))

    lr_var_DEC = tf.Variable(0.001, name='learning_rate_DEC', trainable=False)
    optim_DEC = tf.train.AdamOptimizer(lr_var_DEC)
    kl_loss_step = optim_DEC.minimize(kl_loss, var_list=list(target_vars.values()) + [centers])

    lr_var_diss = tf.Variable(0.002, name='learning_rate_diss', trainable=False)
    optim_diss = tf.train.AdamOptimizer(lr_var_diss)
    # diss_loss_step = optim_diss.minimize(diss_loss, var_list=target_vars.values())
    diss_loss_step = optim_diss.minimize(diss_loss, var_list=[centers])

    # set up session and initialize
    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())

    # restore weights
    if os.path.isdir(weights):
        weights = tf.train.latest_checkpoint(weights)
    logging.info('Restoring weights from {}:'.format(weights))
    logging.info('    Restoring source model:')
    for src, tgt in source_vars.items():
        logging.info('        {:30} -> {:30}'.format(src, tgt.name))
    source_restorer = tf.train.Saver(var_list=source_vars)
    source_restorer.restore(sess, weights)
    logging.info('    Restoring target model:')
    for src, tgt in target_vars.items():
        logging.info('        {:30} -> {:30}'.format(src, tgt.name))
    target_restorer = tf.train.Saver(var_list=target_vars, max_to_keep=10000)
    target_restorer.restore(sess, weights)

    val_restorer = tf.train.Saver(var_list=val_vars)

    # optimization loop (finally)
    output_dir = os.path.join('snapshot', output)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    mapping_losses = deque(maxlen=10)
    adversary_losses = deque(maxlen=10)
    bar = range(iterations)#tqdm(range(iterations))
    # bar.set_description('{} (lr: {:.0e})'.format(output, lr))
    # bar.refresh()
    val_acc = []
    iters = []
    val_iters = list(range(0,200,10)) + list(range(200,iterations+1,100))
    for i in bar:

        kl_val = -1
        diss_loss_v = -1

        source_ims, target_ims = sess.run([source_im_batch, target_im_batch])

        adda_steps = 0#21
        if i == adda_steps:
            while input('Initialize centers and press "c" to continue:') != 'c':
                pass
            data = np.load(centroids_path).item()
            centers_data = data['centers']
            centers_label = data['labels']
            centers_op = tf.assign(centers,centers_data)
            sess.run(centers_op)
        if i>=adda_steps:
            Q_val = sess.run(Q, feed_dict={target_im_batch_h: target_ims})
            P_val = Q_val ** 2
            f = Q_val.sum(axis=0).reshape([1, -1])
            P_val = P_val / f
            s = P_val.sum(axis=1).reshape([-1, 1])
            P_val = P_val / s
            kl_val, _ = sess.run([kl_loss, kl_loss_step], feed_dict={target_im_batch_h: target_ims, P: P_val})
            cluster_assign_v = np.argmax(Q_val, axis=1)
            if len(np.unique(cluster_assign_v)) == n_clusters:
                diss_loss_v, centers_v, _ = sess.run([diss_loss, centers, diss_loss_step])
                print(np.argmax(centers_v, axis=1).flatten())
            else:
                print('Only {} clusters for iteration {}.'.format(len(np.unique(cluster_assign_v)), i))

        weights_ins_v = np.ones(2 * batch_size)
        mapping_loss_val, adversary_loss_val, _, _ = sess.run(
            [mapping_loss, adversary_loss, mapping_step, adversary_step], feed_dict={source_im_batch_h: source_ims,
                                                                                     target_im_batch_h: target_ims,
                                                                                     weights_ins: weights_ins_v})


        mapping_losses.append(mapping_loss_val)
        adversary_losses.append(adversary_loss_val)
        if i % display == 0:
            logging.info('{:20} Mapping: {:10.4f}     (avg: {:10.4f})'
                        '    Adversary: {:10.4f}     (avg: {:10.4f})'
                        '    lk_loss: {:10.4f}'
                        '    diss_liss: {:10.4f}'
                        .format('Iteration {}:'.format(i),
                                mapping_loss_val,
                                np.mean(mapping_losses),
                                adversary_loss_val,
                                np.mean(adversary_losses),
                                kl_val,
                                diss_loss_v))
        if stepsize is not None and (i + 1) % stepsize == 0:
            lr = sess.run(lr_var.assign(lr * 0.1))
            logging.info('Changed learning rate to {:.0e}'.format(lr))
            bar.set_description('{} (lr: {:.0e})'.format(output, lr))
        if (i) % snapshot == 0 or i in val_iters or i+1==adda_steps:
            snapshot_path = target_restorer.save(
                sess, os.path.join(output_dir, output), global_step=i + 1)
            logging.info('Saved snapshot to {}'.format(snapshot_path))

        if i in val_iters:#(i) % 100 == 0:
            print(output_dir)
            weights = tf.train.latest_checkpoint(output_dir)
            logging.info('Evaluating {}'.format(weights))
            val_restorer.restore(sess, weights)
            n_corrects = np.zeros(10)
            n_samples = np.zeros(10)
            for k in range(len(val_dataset)):
                predictions, gt = sess.run([val_out, val_label_batch])
                n_samples[gt[0]] += 1
                if predictions[0] == gt[0]:
                    n_corrects[gt[0]] += 1
            class_acc = n_corrects / n_samples
            acc = n_corrects.sum()/n_samples.sum()
            logging.info('Class-wise accuracy:')
            logging.info( '  '.join(['{:.3f}'.format(x) for x in class_acc]))
            logging.info('Overall accuracy: {}'.format(acc))
            val_acc.append(acc)
            iters.append(i)

    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    main()
