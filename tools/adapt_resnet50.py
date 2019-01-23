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
from adda.data.office31 import office31
from nets.resnet_v1 import resnet_v1_50
from nets import resnet_v1

# import matplotlib.pyplot as plt

slim = tf.contrib.slim


@click.command()
@click.option('--source', default='data/office31:amazon')
@click.option('--target', default='data/office31:dslr')
@click.option('--model', default='resnet_v1_50')
@click.option('--output', default='adapt_resv1_amazon31_dslr10')
@click.option('--gpu', default='0')
@click.option('--iterations', default=10000)
@click.option('--batch_size', default=32)
@click.option('--display', default=10)
@click.option('--lr', default= 0.0002)
@click.option('--stepsize', type=int)
@click.option('--snapshot', default=100)
@click.option('--weights', required=True, default='snapshot/res1_office31_amazon')
@click.option('--solver', default='adam')
@click.option('--adversary', 'adversary_layers', default=[500, 500],
              multiple=True)
@click.option('--adversary_leaky/--adversary_relu', default=True)
@click.option('--seed', type=int)
def main(source, target, model, output,
         gpu, iterations, batch_size, display, lr, stepsize, snapshot, weights,
         solver, adversary_layers, adversary_leaky, seed):

    centroids_path = os.path.join('snapshot',output,'means_31.npy')#'kmeans_centers_mnist_10_usps.npy'
    n_clusters = 31

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

    # setup data
    logging.info('Adapting {} -> {}'.format(source, target))
    ds_source = office31('data/office31', source_split_name)  # ,selected=['bike_helmet'])
    source_split = getattr(ds_source, 'train')
    source_im, source_label = source_split.tf_ops()
    source_im = ds_source.preprocess(source_im, istraining=True)
    source_im_batch, source_label_batch = tf.train.batch([source_im, source_label], batch_size=batch_size)
    selected = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse',
                'mug', 'projector']
    ds_target = office31('data/office31', target_split_name, selected=selected)
    target_split = getattr(ds_target, 'train')
    target_im, target_label = target_split.tf_ops()
    target_im = ds_target.preprocess(target_im, istraining=True)
    target_im_batch, target_label_batch = tf.train.batch([target_im, target_label], batch_size=batch_size)

    ds_s_DEC = office31('data/office31', source_split_name)  # ,selected=['bike_helmet'])
    split_s_DEC = getattr(ds_s_DEC, 'train')
    im_s_DEC, label_s_DEC = split_s_DEC.tf_ops()
    im_s_DEC = ds_s_DEC.preprocess(im_s_DEC, istraining=True)
    im_batch_s_DEC, label_batch_s_DEC = tf.train.batch([im_s_DEC, label_s_DEC], batch_size=batch_size)

    selected = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse',
                'mug', 'projector']
    ds_t_DEC = office31('data/office31', target_split_name, selected=selected)
    split_t_DEC = getattr(ds_t_DEC, 'train')
    im_t_DEC, label_t_DEC = split_t_DEC.tf_ops()
    im_t_DEC = ds_t_DEC.preprocess(im_t_DEC, istraining=True)
    im_batch_t_DEC, label_batch_t_DEC = tf.train.batch([im_t_DEC, label_t_DEC], batch_size=batch_size)

    ds_val = office31('data/office31', target_split_name, selected=selected)
    val_split = getattr(ds_val, 'train')
    val_im, val_label = val_split.tf_ops()
    val_im = ds_source.preprocess(val_im, istraining=False)
    val_im_batch, val_label_batch = tf.train.batch([val_im, val_label], batch_size=1)

    # base network
    source_im_batch_h = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        source_ft, _ = resnet_v1_50(source_im_batch_h, 31, is_training=True, scope='source')
    target_im_batch_h = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        target_ft, _ = resnet_v1_50(target_im_batch_h, 31, is_training=True, scope='target')

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        val_ft, _ = resnet_v1_50(val_im_batch, 31, is_training=False, scope='target', reuse=True)
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
    weights_ins = tf.placeholder(tf.float32, shape=(4*batch_size))
    mapping_loss = tf.losses.sparse_softmax_cross_entropy(
        1 - adversary_label, adversary_logits)
    adversary_loss = tf.losses.sparse_softmax_cross_entropy(
        adversary_label, adversary_logits, weights=weights_ins)

    # kl_loss
    # data = np.load(centroids_path).item()
    # centers = data['centers']
    # centers_label = data['labels']
    centers = tf.Variable(np.zeros((n_clusters,31),dtype=np.float32), dtype=tf.float32, trainable=True)
    assert centers.get_shape()[0] == n_clusters
    assert centers.get_shape()[1] == target_ft.shape[1]
    Q = []
    for i in range(2*batch_size):
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
    P = tf.placeholder(dtype=tf.float32, shape=(2*batch_size, n_clusters))
    # KL divergence
    I = P * tf.log(P / Q)
    kl_loss = tf.reduce_sum(I)/batch_size

    # dissimiliar loss
    centers_prob = tf.contrib.layers.softmax(centers)
    sim_mat = tf.matmul(centers_prob, centers_prob, transpose_b=True)
    diss_loss = tf.sqrt(tf.reduce_sum(tf.square(sim_mat -tf.diag( tf.diag_part(sim_mat) ))))


    # variable collection
    source_vars = adda.util.collect_vars('source')
    target_vars = adda.util.collect_vars('target')
    adversary_vars = adda.util.collect_vars('adversary')

    target_vars_train = []
    layers = ['block4/unit_1/bottleneck_v1/shortcut/weights',
              'block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta',
              'block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma',
              'block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean',
              'block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance',
              'block4/unit_1/bottleneck_v1/conv1/weights',
              'block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta',
              'block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma',
              'block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean',
              'block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance',
              'block4/unit_1/bottleneck_v1/conv2/weights',
              'block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta',
              'block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma',
              'block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean',
              'block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance',
              'block4/unit_1/bottleneck_v1/conv3/weights',
              'block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta',
              'block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma',
              'block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean',
              'block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance',
              'block4/unit_2/bottleneck_v1/conv1/weights',
              'block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta',
              'block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma',
              'block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean',
              'block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance',
              'block4/unit_2/bottleneck_v1/conv2/weights',
              'block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta',
              'block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma',
              'block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean',
              'block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance',
              'block4/unit_2/bottleneck_v1/conv3/weights',
              'block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta',
              'block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma',
              'block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean',
              'block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance',
              'block4/unit_3/bottleneck_v1/conv1/weights',
              'block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta',
              'block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma',
              'block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean',
              'block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance',
              'block4/unit_3/bottleneck_v1/conv2/weights',
              'block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta',
              'block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma',
              'block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean',
              'block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance',
              'block4/unit_3/bottleneck_v1/conv3/weights',
              'block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta',
              'block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma',
              'block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean',
              'block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance',
              'logits/weights',
              'logits/biases']
    for layer in layers:
        target_vars_train.append(target_vars[layer])


    # optimizer
    lr_d = 1e-3
    lr_var_d = tf.Variable(lr_d, name='learning_rate_d', trainable=False)
    lr_t = 1e-5
    lr_var_t = tf.Variable(lr_t, name='learning_rate_t', trainable=False)
    if solver == 'sgd':
        # optimizer = tf.train.MomentumOptimizer(lr_var, 0.9)
        optim_t = tf.train.MomentumOptimizer(1e-5, 0.9)
        optim_d = tf.train.MomentumOptimizer(1e-3, 0.9)
    else:
        # optimizer = tf.train.AdamOptimizer(lr_var, 0.5)
        optim_t = tf.train.AdamOptimizer(lr_var_t, 0.5, 0.9)
        optim_d = tf.train.AdamOptimizer(lr_var_d, 0.5, 0.9)
    mapping_step = optim_t.minimize(
        mapping_loss, var_list=target_vars_train)  # var_list=list(target_vars.values()))
    adversary_step = optim_d.minimize(
        adversary_loss, var_list=list(adversary_vars.values()))

    lr_DEC = 0.0001
    lr_var_DEC = tf.Variable(lr_DEC, name='learning_rate_DEC', trainable=False)
    optim_DEC = tf.train.AdamOptimizer(lr_var_DEC)
    kl_loss_step = optim_DEC.minimize(kl_loss, var_list=target_vars_train + [centers])

    lr_diss = 0.0002
    lr_var_diss = tf.Variable(lr_diss, name='learning_rate_diss', trainable=False)
    optim_diss = tf.train.AdamOptimizer(lr_var_diss)
    diss_loss_step = optim_diss.minimize(diss_loss, var_list=[centers])

    # set up session and initialize
    config = tf.ConfigProto(device_count=dict(GPU=1))
    config = tf.ConfigProto()
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
    val_iters = list(range(0,iterations,10))
    for i in bar:

        kl_vals = -1
        diss_loss_v = -1

        adda_steps = 0
        if i == adda_steps:
            while input('Initialize centers and press "c" to continue:') != 'c':
                pass
            data = np.load(centroids_path).item()
            centers_data = data['centers']
            centers_label = data['labels']
            centers_op = tf.assign(centers,centers_data)
            sess.run(centers_op)
        if i>=adda_steps:
            Q_vals = []
            kl_vals = []
            for s in range(6):
                DEC_ims_s = np.concatenate([sess.run(im_batch_s_DEC), sess.run(im_batch_t_DEC)], axis=0)
                Q_val = sess.run(Q, feed_dict={target_im_batch_h: DEC_ims_s})
                Q_vals.append(Q_val)
                P_val = Q_val ** 2
                f = Q_val.sum(axis=0).reshape([1, -1])
                P_val = P_val / f
                s = P_val.sum(axis=1).reshape([-1, 1])
                P_val = P_val / s
                kl_val, _ = sess.run([kl_loss, kl_loss_step], feed_dict={target_im_batch_h: DEC_ims_s, P: P_val})
                kl_vals.append(kl_val)
            Q_vals = np.concatenate(Q_vals, axis=0)
            kl_vals = np.mean(kl_vals)
            cluster_assign_v = np.argmax(Q_vals, axis=1)
            if len(np.unique(cluster_assign_v)) >= n_clusters - 2:
                diss_loss_v, centers_v, _ = sess.run([diss_loss, centers, diss_loss_step])
                print([ss for ss in range(n_clusters) if ss not in cluster_assign_v])
                print(np.argmax(centers_v, axis=1).flatten())
                # diss_loss_v, _ = sess.run([diss_loss, diss_loss_step], feed_dict={target_im_batch_h: target_ims, cluster_assign: cluster_assign_v})
                # sim_mat_v = sess.run(sim_mat, feed_dict={target_im_batch_h: target_ims, cluster_assign: cluster_assign_v})
                # print(sim_mat_v)
            else:
                print('Only {} clusters for iteration {}.'.format(len(np.unique(cluster_assign_v)), i))
        # print('Number of clusters: {}'.format(len( np.unique(cluster_assign_v) )))
            if (i + 1 - adda_steps) % 50 == 0:  # stepsize is not None and (i + 1) % stepsize == 0:
                lr_DEC = sess.run(lr_var_DEC.assign(lr_DEC * 0.1))
                logging.info('Changed DEC learning rate to {:.0e}'.format(lr_DEC))
                lr_diss = sess.run(lr_var_diss.assign(lr_diss * 0.1))
                logging.info('Changed diss learning rate to {:.0e}'.format(lr_diss))

        source_ims = []
        for k in range(2):
            source_ims.append(sess.run(source_im_batch))
        source_ims = np.concatenate(source_ims, axis=0)
        target_ims = np.concatenate([sess.run(source_im_batch), sess.run(target_im_batch)], axis=0)
        # target_ims = np.concatenate([sess.run(source_im_batch), sess.run(source_im_batch), sess.run(target_im_batch)], axis=0)
        weights_ins_v = np.ones(4 * batch_size)
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
                                kl_vals,
                                diss_loss_v))
            lr_d = sess.run(lr_var_d.assign(lr_d * 0.1))
            logging.info('Changed distriminator learning rate to {:.0e}'.format(lr_d))
            # bar.set_description('{} (lr: {:.0e})'.format(output, lr))
        if (i+1) % 100 == 0:#stepsize is not None and (i + 1) % stepsize == 0:
            lr_t = sess.run(lr_var_t.assign(lr_t * 0.1))
            logging.info('Changed target learning rate to {:.0e}'.format(lr_t))
        if (i) % snapshot == 0 or i in val_iters:
            snapshot_path = target_restorer.save(
                sess, os.path.join(output_dir, output), global_step=i + 1)
            logging.info('Saved snapshot to {}'.format(snapshot_path))

        if i in val_iters:#(i) % 100 == 0:
            print(output_dir)
            n_corrects = np.zeros(31)
            n_samples = np.zeros(31)
            for k in range(len(val_split)):
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
