# DCUDA
This code runs the A31->D10 partial domain adaptation on Office31 for this [paper](https://arxiv.org/abs/1905.13331). Go to the home directory of the project and run the following:

```
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
mkdir data snapshot
```

Please always stay in the project home directory on running the code.

To run the this experiment, you need to download the [Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt) dataset and run:

```
mkdir data/office31
tar -C ./data/office31 -zxvf PATH_TO_OFFICE31_PACKAGE
bash resnet_v1_slim/Download_Checkpoint 
```

First, train a source model on Amazon and evaluate on DSLR:

```
python tools/train_resv1.py data/office31 amazon resnet_v1_50 res1_office31_amazon
python tools/eval_classification_resv1.py  --dataset data/office31 --split dslr --model resnet_v1_50 --weights snapshot/res1_office31_amazon
```

Then start the adaptation process by,

```
python tools/adapt_resnet50.py --source data/office31:amazon --target data/office31:dslr --model resnet_v1_50 --output adapt_resv1_amazon31_dslr10
```

After the program paused, run the following to initialize the centers,

```
python tools/initialize_cluster_centers_target_office_ts.py  
```

then press 'c' to continue. You can evaluate the performance by,

```
python tools/eval_classification_resv1.py  --dataset data/office31 --split dslr --model resnet_v1_50 --weights snapshot/adapt_resv1_amazon31_dslr10
```
