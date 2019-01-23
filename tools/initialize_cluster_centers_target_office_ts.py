import logging
import os
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import adda
from adda.data.office31 import office31
from adda.models.alexnet import AlexNet
from nets.resnet_v1 import resnet_v1_50
from nets import resnet_v1

slim = tf.contrib.slim

dataset_name = 'data/office31'
source_split = 'amazon'
target_split = 'dslr'
model = 'resnet50_v1'
save_path = 'snapshot/adda_resv1_amazon_dslr_DEC_diss_addaf_st_2'
weights_d = 'snapshot/res1_office31_amazon'
batch_size = 25
selected = None
gpu = '0'

os.environ['CUDA_VISIBLE_DEVICES'] = gpu

# dataset = getattr(adda.data.get_dataset(dataset_name, ratios=ratios),split)#, selected=selected, ratios=ratios),split)
# imgs, labels = dataset.tf_ops()
# model_fn = adda.models.get_model_fn(model)
# imgs = adda.models.preprocessing(imgs, model_fn)
# im_batch, label_batch = tf.train.batch(
#         [imgs, labels], batch_size=batch_size)
# ft, _ = model_fn(im_batch, scope='model')

ds_s = office31('data/office31', source_split)  # ,selected=['bike_helmet'])
split_s = getattr(ds_s, 'train')
im_s, label_s = split_s.tf_ops()
im_s = ds_s.preprocess(im_s, istraining=True)
im_batch_s, label_batch_s = tf.train.batch([im_s, label_s], batch_size=batch_size)

selected = ['back_pack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']
ds_t = office31('data/office31', target_split ,selected=selected)
split_t = getattr(ds_t, 'train')
im_t, label_t = split_t.tf_ops()
im_t = ds_t.preprocess(im_t, istraining=True)
im_batch_t, label_batch_t = tf.train.batch([im_t, label_t], batch_size=batch_size)

im_batch_h = tf.placeholder(tf.float32, (None, 224,224,3))
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    ft, _ = resnet_v1_50(im_batch_h, 31, is_training=True, scope='model')




vars = adda.util.collect_vars('model')

# set up session and initialize
config = tf.ConfigProto(device_count=dict(GPU=1))
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
sess.run(tf.global_variables_initializer())

# restore weights
if os.path.isdir(weights_d):
    weights = tf.train.latest_checkpoint(weights_d)
logging.info('Restoring weights from {}:'.format(weights))
logging.info('    Restoring source model:')
for src, tgt in vars.items():
    logging.info('        {:30} -> {:30}'.format(src, tgt.name))
logging.info('    Restoring target model:')
for src, tgt in vars.items():
    logging.info('        {:30} -> {:30}'.format(src, tgt.name))
vars_restorer = tf.train.Saver(var_list=vars, max_to_keep=10000)
vars_restorer.restore(sess, weights)

ft_bs_s = []
label_bs_s = []
label_bs_p_s = []
for i in range(20):
    ims_v, label_v = sess.run([im_batch_s, label_batch_s])
    ft_v  = sess.run(ft, feed_dict={im_batch_h: ims_v})
    label_p = np.argmax(ft_v, axis=1).reshape((-1,))
    ft_bs_s.append(ft_v)
    label_bs_s.append(label_v)
    label_bs_p_s.append(label_p)
ft_bs_s = np.concatenate(ft_bs_s, axis=0)
label_bs_s = np.concatenate(label_bs_s, axis=0)
label_bs_p_s = np.concatenate(label_bs_p_s, axis=0)

ft_bs_t = []
label_bs_t = []
label_bs_p_t = []
for i in range(20):
    ims_v, label_v = sess.run([im_batch_t, label_batch_t])
    ft_v  = sess.run(ft, feed_dict={im_batch_h: ims_v})
    label_p = np.argmax(ft_v, axis=1).reshape((-1,))
    ft_bs_t.append(ft_v)
    label_bs_t.append(label_v)
    label_bs_p_t.append(label_p)
ft_bs_t = np.concatenate(ft_bs_t, axis=0)
label_bs_t = np.concatenate(label_bs_t, axis=0)
label_bs_p_t = np.concatenate(label_bs_p_t, axis=0)

n_clusters=31
# print( np.unique(label_bs,return_counts=True)[1] )
# for i in range(n_clusters):
#     num_p = sum(label_bs_p == i)
#     num_gt = sum(label_bs == i)
#     print('{} gt:{}, pred:{}'.format(i,num_gt,num_p))

# kmeans = KMeans(n_clusters=n_clusters, verbose=1, random_state=20).fit(ft_bs)
# label_km = kmeans.labels_
# cluster_centers = kmeans.cluster_centers_

cluster_centers = []
for l in range(31):
    c = np.mean( ft_bs_s[label_bs_p_s==l,:],axis=0 )
    if l in label_bs_p_t:
        c_t = np.mean( ft_bs_t[label_bs_p_t==l,:],axis=0 )
        c = 0.9*c + 0.1*c_t
    cluster_centers.append(c)
cluster_centers = np.stack(cluster_centers,axis=0)
# label_km = label_bs_p

# centers_label = []
# for i in range(n_clusters):
#     ls = label_bs_p[label_km==i]
#     l_u, count = np.unique(ls, return_counts=True)
#     ind = np.argmax(count)
#     centers_label.append(l_u[ind])
# centers_label = np.array(centers_label)
centers_label = np.array(range(n_clusters))

np.save( os.path.join(save_path, 'kmeans_31.npy'), {'centers': cluster_centers, 'labels': centers_label} )

ft_em_all = TSNE(n_components=2,verbose=1,perplexity=50).fit_transform(np.concatenate([cluster_centers, ft_bs_s, ft_bs_t], axis=0))
centers_em = ft_em_all[:n_clusters]
ft_em_s = ft_em_all[n_clusters:(n_clusters+len(ft_bs_s))]
ft_em_t = ft_em_all[(n_clusters+len(ft_bs_s)):]


plt.figure()
colors = cm.rainbow(np.linspace(0,1,31))
colors_dict = {}
for l, c in zip(range(31),colors):
    colors_dict[l] = c
# h=plt.scatter(ft_em[:,0],ft_em[:,1],c=label_bs)
ls = []
hs = []
for l, c in enumerate(colors):#zip(selected,colors):
    h = plt.scatter(ft_em_s[label_bs_s==l,0],ft_em_s[label_bs_s==l,1],color=c,s=2)
    plt.scatter(ft_em_t[label_bs_t == l, 0], ft_em_t[label_bs_t == l, 1], color=c, s=2)
    hs.append(h)
    ls.append(str(l))
plt.title('Classes')
plt.figure()
for l, c in enumerate(colors):#zip(selected,colors):
    h = plt.scatter(ft_em_s[label_bs_p_s==l,0],ft_em_s[label_bs_p_s==l,1],color=c,s=2)
    plt.scatter(ft_em_t[label_bs_p_t == l, 0], ft_em_t[label_bs_p_t == l, 1], color=c, s=2)
    plt.scatter(centers_em[l,0], centers_em[l,1],color='b',marker='x',s=20)
    hs.append(h)
    ls.append(str(l))
plt.title('Clusters')
plt.figure()
colors = cm.rainbow(np.linspace(0,1,n_clusters))
# plt.scatter(ft_em[:,0],ft_em[:,1],c=label_km,cmap=matplotlib.colors.ListedColormap(colors), s=2)
for l, c in enumerate(colors):
    h = plt.scatter(ft_em_s[label_bs_s == l, 0], ft_em_s[label_bs_s == l, 1], color=c, s=2)
plt.title('Source')
plt.figure()
for l, c in enumerate(colors):
    h = plt.scatter(ft_em_t[label_bs_t == l, 0], ft_em_t[label_bs_t == l, 1], color=c, s=2)
plt.title('Target')
plt.show()

