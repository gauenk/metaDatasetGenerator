# metaDatasetGenerator

## Install on HELPS computer

### Part 1: Linking with Caffe

While not strictly required, this repo should work with caffe.

First, add the following line to the bottom of your "~/.bashrc" file.

```Shell
export PYTHONPATH="/opt/caffe-fast-rcnn/python"
```

Then run:

```Shell
source ~/.bashrc
```

To install dependancies for caffe run:

```Shell
pip2 install --upgrade pip --user
pip2 install easydict --user
cd /opt/caffe-fast-rcnn/python
cat requirements.txt | xargs pip2 install --user
```

You should check the caffe install by running the following in your shell:

```Shell
username@computerName:$ python
>>> import caffe
>>>
```

### Part 2: Linking the github repo

Then run the following github information:

```Shell
cd ~/
git clone https://github.com/gauenk/metaDatasetGenerator.git
cd ./metaDatasetGenerator/lib
make
cd ../
./tools/add_helps_symlinks.sh
```

To verify the complete installation of the github repo type:

```Shell
./experiments/scripts/imdbReport.sh pascal_voc_2007 train
```

It may complain the filepath for pascal_voc_2007 is set incorrectly. You may need to modify the config file to tell to load `ymlDatasets/helps/` and not `ymlDatasets/username`. This is set at the top of `./lib/core/config.py` (around line #35). Make sure it reads:

```Python
__C.PATH_YMLDATASETS = "helps"
```

## Overview

This repo generates information about the paper written by the CAM2 team in 2018. The repo offers functionality for:

1. generating sample datasets from a mixture of original datasets
   -> saving (&loading) these sample datasets to (&from) file
2. computing properties about each sample dataset
   -> seperability from the original dataset repo with
      -> HOG + SVM
      -> Deep Learning models
   -> similarity of vae representation (interpret them as codes)
   -> seperability of the annotations (uses information about annotations)
   -> saving this report to file
3. train a deep learning model on a sample dataset
   -> save the model weights periodically
4. test a deep learning model on a dataset
   -> handle the separation of training and testing data
   -> what size of datasets should be used for testing?
5. computing properties about each sample dataset

What questions do we aim to answer?
1. What is the relationship between properties of sample dataset images and the variance of the model (x-dataset generalization)?
2. How can we interpret this relationship with respect to standard interpretations of active learning?
3. What is the relationship between properties of sample dataset images and the interpretability of the model?
4. What is the relationship between b, the sample dataset annotations, and the observed model outcomes?


# Directories

## [./lib/core](./lib/core/)
## [./lib/datasets](./lib/datasets/)

# TODO

-> create a sample dataset object
   -> it takes in imdbs (image data bases) as it's input with the mixture of each imdb defined
   -> it is a child of the "imdb" class; e.g. the interface is the same
   -> it can grow
   -> it can save the ids used at each step of growth

-> create modules (a folder in `./lib`) where we compute intrinsic information about datasets
      -> this is an extension of the work over the previous semester
      -> this includes:
      	 -> the "name that dataset" game
	 -> the annotation distribution information (metrics M1 & M2, ...)

-> create modules (a folder in `./lib`) to compute testing information about a given ".caffemodel" file

## Usage


The first requirement is to generate the mixture datasets in the `./data/mixtureDatasets/` folder. This can be done by running the `genMixData.sh` script. For example:
```
Usage: ./experiments/scripts/genMixData.sh START_INDEX END_INDEX REPEAT
```

An example useage is:
```Shell
./experiments/scripts/genMixData.sh 2 3 4
```

The "START" and "END" "_INDEX" indicates the range for how many datasets are mixed together. For example, if the numbers were "2" and "4" the script would generate all mixtures of 2, 3, and 4 datasets.

## Misc

To see information about an imdb such as number of annotations, number of images, and approx. memory usage use the script:

```Shell
experiments/scripts/imdbReport.sh pascal_voc_2007 train
```
