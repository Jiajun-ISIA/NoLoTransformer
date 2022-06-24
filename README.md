# NoLoTransformer

## What can I find here?

This repository contains all code and implementations used in:

```
A Noise-robust Locality Transformer for Fine-grained Food Image Retrieval
```
accepted to MIPR 2022

### Requirements:

* PyTorch 1.2.0+ & Faiss-Gpu
* Python 3.6+
* pretrainedmodels, torchvision 0.3.0+

An exemplary setup of a virtual environment containing everything needed:
```
(1) wget  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
(2) bash Miniconda3-latest-Linux-x86_64.sh (say yes to append path to bashrc)
(3) source .bashrc
(4) conda create -n DL python=3.6
(5) conda activate DL
(6) conda install matplotlib scipy scikit-learn scikit-image tqdm pandas pillow
(7) conda install pytorch torchvision faiss-gpu cudatoolkit=10.0 -c pytorch
(8) pip install wandb pretrainedmodels
(9) Run the scripts!
```

### Datasets:
Data for
* ETH Food-101, Vireo Food-172, ISIA Food-200


* For ETH Food-101:
```
food101
└───images
|    └───apple_pie
|           │   134.jpg
|    ...
```

Assuming your folder is placed in e.g. `<$datapath/food>`, pass `$datapath` as input to `--source`.

### Training:
Training is done by using `main.py` and setting the respective flags, all of which are listed and explained in `parameters.py`.

**A basic sample run using the best parameters would like this**:

```
CUDA_VISIBLE_DEVICES=0 python main.py --m_loss margin --seed 0 --bs 112  --samples_per_class 2 --arch ours_model --source ../dataset_food --n_epochs 100 --lr 1e-6 --embed_dim 128 --evaluate_on_gpu  --dataset food101

```
## Paper
If you find this work useful, please consider citing:
```
@InProceedings{Jiajun2022,
  author       = "Jiajun Song and Weiqing Min and Yuxin Liu and Zhuo Li and Shuqiang Jiang and Yong Rui",
  title        = "A Noise-robust Locality Transformer for Fine-grained Food Image Retrieval",
  booktitle    = "Fifth IEEE International Conference on Multimedia Information Processing and Retrieval (MIPR 2022)",
  year         = "2022",
}
```
