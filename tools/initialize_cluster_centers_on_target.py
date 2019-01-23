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

dataset_name = 'mnist'
split = 'test'
model = 'lenet'
save_path = 'snapshot/adapt_lenet_svhn_mnist'
weights_d = 'snapshot/lenet_svhn'
batch_size = 128
selected = list(range(10))
ratios = None#np.linspace(1,0.3,len(selected))
gpu = '1'

os.environ['CUDA_VISIBLE_DEVICES'] = gpu


dataset = getattr(adda.data.get_dataset(dataset_name, selected=selected, ratios=None),split)#, selected=selected, ratios=ratios),split)
imgs, labels = dataset.tf_ops()
model_fn = adda.models.get_model_fn(model)
imgs = adda.models.preprocessing(imgs, model_fn)
im_batch, label_batch = tf.train.batch(
        [imgs, labels], batch_size=batch_size)
ft, _ = model_fn(im_batch, scope='model', n_classes=len(selected))

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

ft_bs = []
label_bs = []
label_bs_p = []
for i in range(20):
    ft_v, label_v = sess.run([ft, label_batch])
    label_p = np.argmax(ft_v, axis=1).reshape((-1,))
    ft_bs.append(ft_v)
    label_bs.append(label_v)
    label_bs_p.append(label_p)
ft_bs = np.concatenate(ft_bs, axis=0)
label_bs = np.concatenate(label_bs, axis=0)
label_bs_p = np.concatenate(label_bs_p, axis=0)

print( np.unique(label_bs,return_counts=True)[1] )

n_clusters=len(selected)
# kmeans = KMeans(n_clusters=n_clusters, verbose=1, random_state=20).fit(ft_bs)
# label_km = kmeans.labels_
# cluster_centers = kmeans.cluster_centers_

cluster_centers = []
for l in range(n_clusters):
    if sum(label_bs_p==l) == 0:
        print('No instance: {}'.format(l))
        c = np.zeros(ft_bs.shape[1])
    else:
        c = np.mean( ft_bs[label_bs_p==l,:],axis=0 )
    cluster_centers.append(c)
cluster_centers = np.stack(cluster_centers,axis=0)
label_km = label_bs_p

# centers_label = []
# for i in range(n_clusters):
#     ls = label_bs_p[label_km==i]
#     l_u, count = np.unique(ls, return_counts=True)
#     ind = np.argmax(count)
#     centers_label.append(l_u[ind])
# centers_label = np.array(centers_label)
centers_label = np.array(list(range(n_clusters)))

np.save( os.path.join(save_path, 'means_10.npy'), {'centers': cluster_centers, 'labels': centers_label} )

ft_em_all = TSNE(n_components=2,verbose=1).fit_transform(np.concatenate([cluster_centers, ft_bs],axis=0))
centers_em = ft_em_all[:n_clusters]
ft_em = ft_em_all[n_clusters:]


fig,ax = plt.subplots(1)
colors = cm.rainbow(np.linspace(0,1,len(selected)))
colors_dict = {}
for l, c in zip(selected,colors):
    colors_dict[l] = c
# h=plt.scatter(ft_em[:,0],ft_em[:,1],c=label_bs)
ls = []
hs = []
for l, c in zip(range(len(selected)),colors):
    h = plt.scatter(ft_em[label_bs==l,0],ft_em[label_bs==l,1],color=c,s=2)
    hs.append(h)
    ls.append(str(selected[l]))
# for i in range(n_clusters):
#     # plt.scatter(ft_em[label_km==i,0],ft_em[label_km==i,1],color=colors_dict[centers_label[i]],s=2)
#     plt.scatter(centers_em[i,0], centers_em[i,1],color='b',marker='x',s=20)
# plt.legend(hs,ls, loc='upper right')
plt.title('Classes')
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.figure()
# colors = cm.rainbow(np.linspace(0,1,n_clusters))
# plt.scatter(ft_em[:,0],ft_em[:,1],c=label_km,cmap=matplotlib.colors.ListedColormap(colors))
for i in range(n_clusters):
    plt.scatter(ft_em[label_km==i,0],ft_em[label_km==i,1],color=colors[i],s=2)
    plt.scatter(centers_em[i,0], centers_em[i,1],color='b',marker='x',s=20)
plt.title('Clusters')
plt.legend()
plt.show()

