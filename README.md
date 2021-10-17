# When Does Self-supervision Improve Few-shot Learning? - A Reproducibility Report

<img src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />  <a href="https://wandb.ai/meta-learners/projects" alt="W&B Dashboard">  <img src="https://img.shields.io/badge/WandB-Dashboard-gold.svg" /></a> <a href="https://dagshub.com/arjun2000ashok/FSL-SSL" alt="DAGsHub Dashboard"><img src="https://img.shields.io/badge/DAGsHub-Project-blue.svg" /></a>



A reproduction of the paper [When Does Self-supervision Improve Few-shot Learning? (ECCV 2020)](https://arxiv.org/abs/1910.03560). We also have some interesting results beyond the paper. The authors' official repository is [here](https://github.com/cvl-umass/fsl_ssl).


As part of the [**ML Reproducibility Challenge Spring 2021**](https://paperswithcode.com/rc2020).

**All the hyperparameters** obtained from our sweeps are also reported in section 7 below.

# Table of contents

**Please click each of the sections below to jump to its contents.**

1. [Installation Instructions](#installation)
2. [Datasets](#datasets)
3. [Training & Inference](#training)
     
     3.1. [General Instructions](#general)

     3.2. [Examples](#examples)
4. [Domain selection](#domain)
5. [Conducting a sweep](#sweep)
6. [Pretrained models](#pretrained)
7. [Hyperparameters](#hyperparams)

<div id='installation' />

# 1. Installation Instructions

Execute this to install all the dependencies:

```pip install -r requirements.txt```

<div id='datasets' />

# 2. Datasets

In total, we support a total of 10 datasets for training and inference. Apart from this, we support one more dataset for the domain selection experiments.

The datasets, details, download links, location are below:

| Dataset          | Download Link                                              | Extraction Location               |
| ---------------- | ---------------------------------------------------------- | --------------------------------- |
| CUB-birds        | https://www.kaggle.com/tarunkr/caltech-birds-2011-dataset  | `filelists/CUB/images`          |
| VGG flowers      | https://www.kaggle.com/arjun2000ashok/vggflowers/          | `filelists/flowers/images`      |
| Stanford Cars    | https://www.kaggle.com/hassiahk/stanford-cars-dataset-full | `filelists/cars/images`         |
| Stanford Dogs    | https://www.kaggle.com/jessicali9530/stanford-dogs-dataset | `filelists/dogs/images`         |
| FGVC - Aircrafts | https://www.kaggle.com/seryouxblaster764/fgvc-aircraft     | `filelists/aircrafts/images`    |
| MiniImageNet     | https://www.kaggle.com/arjunashok33/miniimagenet           | `filelists/miniImagenet/images` |


If you are using the Aircrafts dataset, you should run `utils/crop_aircrafts.py`. It will be crop the bottom 20 rows of the image which contain the image credits. In the original repository, this is done in the code. To optimize that, this script has been provided.

<br>

For the cross-domain experiments, we list the datasets here for convinience. These links are taken from the [CDFSL benchmark repository](https://github.com/IBM/cdfsl-benchmark).


| Dataset     | Download Link                                    | Extraction Location              |
| ----------- | ------------------------------------------------ | -------------------------------- |
| ChestX      | https://www.kaggle.com/nih-chest-xrays/data      | `filelists/ChestX/images`      |
| ISIC        | https://challenge.isic-archive.com - **NOTE:** needs login | `filelists/ISIC/images`        |
| EuroSAT     | http://madm.dfki.de/files/sentinel/EuroSAT.zip   | `filelists/EuroSAT/images`     |
| CropDisease | https://www.kaggle.com/saroz014/plant-disease/   | `filelists/CropDisease/images` |

<div id='training' />

# 3. Training & Inference

<div id='general' />

## 3.1 General Instructions

**All scripts must be executed from the `root` folder**

The `train.py` file is used for training, validation and testing.

It trains the few-shot model for a fixed number of episodes, with periodic evalution on the validation set, followed by testing on the test set.


Note that all the results reported are based on training for a fixed number of epochs, and then evaluating using the best model found using the validation set.

Please see `utils/io_utils.py` for all the arguments and their default values. Here are some sufficient examples:

`python train.py --help` will print the help for all the necessary arguments:

```
usage: train.py [-h] [--dataset DATASET] [--model MODEL] [--method METHOD]
                [--train_n_way TRAIN_N_WAY] [--test_n_way TEST_N_WAY]
                [--n_shot N_SHOT] [--train_aug [TRAIN_AUG]]
                [--jigsaw [JIGSAW]] [--lbda LBDA] [--lr LR]
                [--optimization OPTIMIZATION] [--loadfile LOADFILE]
                [--finetune] [--random] [--n_query N_QUERY]
                [--image_size IMAGE_SIZE] [--debug] [--json_seed JSON_SEED]
                [--date DATE] [--rotation [ROTATION]] [--grey]
                [--low_res [LOW_RES]] [--firstk FIRSTK] [--testiter TESTITER]
                [--wd WD] [--bs BS] [--iterations ITERATIONS] [--useVal]
                [--scheduler [SCHEDULER]] [--lbda_jigsaw LBDA_JIGSAW]
                [--lbda_rotation LBDA_ROTATION] [--pretrain [PRETRAIN]]
                [--dataset_unlabel DATASET_UNLABEL]
                [--dataset_unlabel_percentage DATASET_UNLABEL_PERCENTAGE]
                [--dataset_percentage DATASET_PERCENTAGE] [--bn_type BN_TYPE]
                [--test_bs TEST_BS] [--split SPLIT] [--save_iter SAVE_ITER]
                [--adaptation] [--device DEVICE] [--seed SEED] [--amp [AMP]]
                [--num_classes NUM_CLASSES] [--save_freq SAVE_FREQ]
                [--start_epoch START_EPOCH] [--stop_epoch STOP_EPOCH]
                [--resume] [--warmup] [--eval_interval EVAL_INTERVAL]
                [--run_name RUN_NAME] [--run_id RUN_ID]
                [--semi_sup [SEMI_SUP]] [--sup_ratio SUP_RATIO]
                [--only_test [ONLY_TEST]] [--project PROJECT]
                [--save_model [SAVE_MODEL]] [--demo [DEMO]]
                [--only_train [ONLY_TRAIN]] [--sweep [SWEEP]]
```

`--device` is used to specify the GPU device.
`--seed` is used to specify the seed. The default 
`--train_aug` is used to specify if there should be data augmentation. All results in the paper and the report are done with data augmentation. It is by default `True`.
`--stop_epoch` is used the number of epochs. It is recommended to run the small dataset models for 500 epochs, and the miniImageNet models for 700 epochs. The best model picked by validation will be evaluated at the end.
`--lr` is the learning rate. 

`--loadfile` can be used to provide a path for loading a pretrained model. Note that the model should be of the same architecture as given in the `--model` argument. 


`--only_train` can be used to only train the models, and stop before testing them.
`--only_test` can be used to only test the models. Note that a model path needs to be provided if you are testing a pretrained model.

`--resume` can be used to resume a run. If `--resume` is provided, the `--run_id` must also be provided to resume the corresponding W&B run. By default, as `line 362` of `train.py` indicates, the last model will be retrieved from W&B automatically, and loaded. Note that I have saved the epoch also, and hence the `--start_epoch` will be automatically set. The `--stop_epoch` must be provided in all cases.

`--bn_type` can be set to `1`, `2` or `3` to set the respective type of batch norm used.
  
`NUM_WORKERS` for all the dataloaders can be set at `config/configs.py`.


<div id='examples' />

## 3.2 Examples

**NOTE**: For the exact commands used, you can refer to each run in the respective project in the W&B dashboard. We provide a detailed list of all the possible configurations below. 
  
### Supervised training

For training a ProtoNet 5-way 5-shot model on CUB dataset with resnet18 and image size 224, for 600 epochs (LR=0.01):

`python train.py --dataset=CUB --model=resnet18 --method=protonet --train_n_way=5 --test_n_way=5 --n_shot=5 --train_aug --stop_epoch=600 --lr=0.01`.

For Conv4 architecture and image size 84, use `--image_size=84 --model=Conv4`.

### With Self-Supervised learning (SSL)


For training with jigsaw with a specific loss weightage, use `--jigsaw --lbda=LBDA` where `LBDA` is substituted by the weightage.

For training with rotation with a specific loss weightage, use `--rotation --lbda=LBDA` where `LBDA` is substituted by the weightage.

For training with both jigsaw and rotation, use `--rotation --jigsaw --lbda_jigsaw=LBDA_JIGSAW --lbda_rotation=LBDA_ROTATION` where `LBDA_JIGSAW` and `LBDA_ROTATION` is substituted by the weightage of jigsaw and rotation in the loss functions respectively. The supervised loss in this case will be automatically set to `1 - LBDA_JIGSAW - LBDA_ROTATION`.


### Using other datasets for SSL


For using other datasets for self-supervised learning, `--dataset_unlabel=DATASET` must be provided. Note that in this case `--jigsaw --lbda=X` or `--rotation --lbda=X` or `--rotation --jigsaw --lbda_jigsaw=LBDA_JIGSAW --lbda_rotation=LBDA_ROTATION` must also be provided as self-supervision is used.

To use only a fraction of the other dataset for self-supervised learning, **along with the above command**, `--dataset_unlabel_percentage` can be set to `20, 40, 60 or 80`.

To use only a fraction of the other dataset for **supervised** learning, `--dataset_percentage` can be set to `20, 40, 60 or 80`.

Both of the above commands will use the respective file from `filelists/{DATASET}`.

### For testing on CDFSL datasets

For this, use `cdfsl_test.py`. By default, I have provided the corresponding table name from W&B. As I have saved all the checkpoints in W&B, it will automatically download them, test on all the 4 CDFSL datasets and return the results.  


 

<div id='domain' />

# 4. Domain selection

We have implemented the domain selection algorithm from scratch. 

To conduct domain selection, the [open images dataset](https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations) train set (**513 GB**) and [INat dataset](https://github.com/visipedia/inat_comp/tree/master/2021) train set (**224 GB**) needs to be downloaded and put at `filelists/open-images/images` and `filelists/inat/images` respectively.

As we could not afford storage for such huge datasets, we used the validation datasets instead (found in the same URLs).

After that, `python domain/importance_weights.py` should be run with arguments `--positive_json` indicating the dataset for which the images need to be selected (paths of the json files of CUB/aircrafts/cars/dogs/flowers), and `--save_path=SAVE_PATH.json` indicating the save path of the `json` containing a list of the respective selected images for the dataset.

This will get the image features from an ImageNet-pretrained ResNet-101, train a logistic regression model with the positive images as the dataset images and negative images from the two new datasets, and select the optimal images for SSL.

An example is below:

`python domain/importance_weights.py --positive_json=filelists/CUB/base.json --save_path=filelists/CUB/selected_domain_images.json` 

After this, the usual training can be done with `dataset_unlabel=filelists/CUB/selected_domain_images.json`.

 


<div id='sweep' />

# 5. Conducting a Sweep

To conduct a sweep in W&B, first create a `yaml` file with the arguments. Examples are provided in the `sweeps`
directory. Note that `save_model=0` can be set here, as we only need the best configuration found from `100 epochs` (or a higher number), and we should re-run with the best configuration.

After creating a `yaml` file, run `python sweep.py --yaml=YAML_PATH --count=x --project=WANDB_PROJECT --gpu=DEVICE_ID` where

* `--yaml` is the `yaml` file path
* `--count` is the sweep count if you are using random search
* `--project` is the w&b project to conduct the sweep in
* `--gpu` is the device ID.


 

<div id='pretrained' />

# 6. Pretrained models

The number of runs are very high in our project, and the total size of all our models comes to **111 GB**. Hence we have kept all the pretrained models in the W&B runs.

The pretrained models for all the runs - both the last model and the best model, are in the respective runs in the projects denoted by the table number in the [W&B dashboard](https://wandb.ai/meta-learners/projects).


You can download the `best_model.tar` from the `files` section in the W&B run, and perform inference in the following manner:

1. Use the same command given in the W&B run, with `--device` altered to match your device.
2. Add `--only_test=True`
3. Add `--loadfile=PATH` where PATH denotes the path where you can downloaded the model.

<div id='hyperparams' />

# 7. Hyperparameters

| Dataset | Config | Way | Architecture | Learning Rate | Batch Norm Mode | Alpha | Alpha\_Jigsaw, Alpha\_Rotation |
| ------- | ------ | --- | ------------ | ------------- | --------------- | ----- | ------------------------------ |
| MIN     | plain  | 5   | Res          | 0.0002        | 1               | \-    |                                |
| MIN     | jig    | 5   | Res          | 0.0002        | 2               | 0.6   |                                |
| MIN     | rot    | 5   | Res          | 0.0004        | 2               | 0.6   |                                |
| MIN     | jigrot | 5   | Res          | 0.0003        | 2               |       | (0.4, 0.5)                     |
|         |        |     |              |               |                 |       |                                |
| MIN     | plain  | 5   | Conv         | 0.003         | 1               | \-    |                                |
| MIN     | jig    | 5   | Conv         | 0.002         | 2               | 0.35  |                                |
| MIN     | rot    | 5   | Conv         | 0.003         | 2               | 0.4   |                                |
|         |        |     |              |               |                 |       |                                |
| MIN     | plain  | 20  | Res          | 0.0001        | 1               | \-    |                                |
| MIN     | jig    | 20  | Res          | 0.001         | 2               | 0.5   |                                |
| MIN     | rot    | 20  | Res          | 0.002         | 2               | 0.44  |                                |
|         |        |     |              |               |                 |       |                                |
| CUB     | plain  | 5   | Res          | 0.0015        | 1               | \-    |                                |
| CUB     | rot    | 5   | Res          | 0.0023        | 2               | 0.65  |                                |
| CUB     | jig    | 5   | Res          | 0.0018        | 2               | 0.33  |                                |
| CUB     | jigrot | 5   | Res          | 0.0009        | 2               |       | (0.55,0.64)                    |
|         |        |     |              |               |                 |       |                                |
| CUB     | plain  | 5   | Conv         | 0.015         | 1               | \-    |                                |
| CUB     | rot    | 5   | Conv         | 0.005         | 2               | 0.3   |                                |
| CUB     | jig    | 5   | Conv         | 0.003         | 2               | 0.38  |                                |
|         |        |     |              |               |                 |       |                                |
| Cars    | jig    | 5   | Conv         | 0.004         | 2               | 0.52  |                                |
| Cars    | rot    | 5   | conv         | 0.016         | 2               | 0.24  |                                |
| Cars    | plain  | 5   | conv         | 0.01          | 1               | \-    |                                |
|         |        |     |              |               |                 |       |                                |
| Cars    | plain  | 5   | Res          | 0.0021        | 1               | \-    |                                |
| Cars    | rot    | 5   | Res          | 0.0038        | 2               | 0.5   |                                |
| Cars    | jig    | 5   | Res          | 0.0035        | 2               | 0.38  |                                |
| Cars    | jigrot | 5   | Res          | 0.0012        | 2               |       | (0.35,0.43)                    |

**NOTE:**

`jig` refers to jigsaw, `rot` refers to rotation, `jigrot` refers to jigsaw+rotation.								

`Conv` refers to the conv-4 architecture, `Res` refers to resnet-18.												

`CUB`, `cars`, `MIN` refer to the respective datasets.																

We always run for a large number of epochs - **600** for the small datasets was found to be sufficient, and **800** for miniImageNet. After this, we pick the model that performed best on the validation set, and evaluate it on the test set.																										
