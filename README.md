# DCUDA
This code runs two domain adaptation experiments in the paper: SVHN->MNIST (balanced  domain adaptation) and A31->D10 (Partial domain adaptation on Office31). Go to the home directory of the project and run the following:

```
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
mkdir data
```

Please always stay in the project home directory on running the code.

## SVHN->MNIST
First, train a svhn source model using lenet and evaluate it on mnist.

```
python tools/train_lenet.py svhn train lenet lenet_svhn
python tools/eval_classification_lenet.py mnist test lenet snapshot/lenet_svhn
```

Then start the adaptation process by,

```
python tools/adapt_lenet.py svhn:train mnist:train lenet adapt_lenet_svhn_mnist
```

After the program paused, run the following to initialize the centers,

```
python tools/initialize_cluster_centers_on_target.py 
```

then press 'c' to continue. You can evaluate the performance by,

```
python tools/eval_classification_lenet.py mnist test lenet snapshot/adapt_lenet_svhn_mnist
```


## A31->D10 on Office31

To run the this experiment, you need to download the [Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt) dataset and run:

```
mkdir data/office31
tar -C ./data/office31 -zxvf PATH_TO_OFFICE31_PACKAGE
bash resnet_v1_slim/Download_Checkpoint 
```

First, train a source model on Amazon and evaluate on DSLR:

```
python tools/train_resv1.py data/office31 amazon resnet_v1_50 res1_office31_amazon
python tools/eval_classification_resv1.py  data/office31 dslr resnet_v1_50 snapshot/res1_office31_amazon
```

Then start the adaptation process by,

```
python tools/adapt_resnet50.py data/office31:amazon data/office31:dslr resnet_v1_50 adapt_resv1_amazon31_dslr10
```

After the program paused, run the following to initialize the centers,

```
python tools/initialize_cluster_centers_target_office_ts.py  
```

then press 'c' to continue. You can evaluate the performance by,

```
python tools/eval_classification_resv1.py  data/office31 dslr resnet_v1_50 snapshot/adapt_resv1_amazon31_dslr10
```
