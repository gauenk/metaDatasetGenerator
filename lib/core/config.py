# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Fast R-CNN config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from core.config import cfg
cfg = __C

#
# Dataset options
#
__C.DATASETS = edict()
cfgData = __C.DATASETS
__C.DATASETS.EXP_DATASET = ""
__C.DATASETS.PATH_ROOT = ""
__C.DATASETS.PATH_TO_IMAGES = ""
__C.DATASETS.PATH_TO_ANNOTATIONS = ""
__C.DATASETS.PATH_TO_IMAGESETS = ""
__C.DATASETS.PATH_TO_RESULTS = ""
__C.DATASETS.CLASSES = ""
__C.DATASETS.COMPID = "defaultCompID"
__C.DATASETS.IMAGE_TYPE = ""
__C.DATASETS.ANNOTATION_TYPE = ""
__C.DATASETS.PARSE_ANNOTATION_REGEX = None
__C.DATASETS.CONVERT_TO_PERSON = None
__C.DATASETS.IMAGE_INDEX_TO_IMAGE_PATH = None
__C.DATASETS.ONLY_PERSON = False
__C.DATASETS.USE_IMAGE_SET = None
__C.DATASETS.CONVERT_ID_TO_CLS_FILE = None
__C.DATASETS.USE_IMAGE_SET = None
__C.DATASETS.CONVERT_ID_TO_CLS_FILE = None
__C.DATASETS.ONLY_PERSON = False

#
# Training options
#

__C.TRAIN = edict()
__C.PATH_YMLDATASETS = "helps"
#__C.PATH_YMLDATASETS = "gauenk"
__C.PATH_MIXTURE_DATASETS = "./lib/datasets/mixtureDatasets/"

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)
#__C.TRAIN.SCALES = (64,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000
#__C.TRAIN.MAX_SIZE = 64

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 2

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 500

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX = ''

# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
__C.TRAIN.USE_PREFETCH = False

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.
__C.TRAIN.ASPECT_GROUPING = True

#
# Training (object detection) options
#

__C.TRAIN.OBJ_DET = edict()

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.OBJ_DET.BATCH_SIZE = 128

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.OBJ_DET.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.OBJ_DET.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.OBJ_DET.BG_THRESH_HI = 0.5
__C.TRAIN.OBJ_DET.BG_THRESH_LO = 0.1

# Train bounding-box regressors
__C.TRAIN.OBJ_DET.BBOX_REG = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.OBJ_DET.BBOX_THRESH = 0.5

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.OBJ_DET.BBOX_NORMALIZE_TARGETS = True
# Deprecated (inside weights)
__C.TRAIN.OBJ_DET.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.OBJ_DET.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
__C.TRAIN.OBJ_DET.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.OBJ_DET.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# Train using these proposals
__C.TRAIN.OBJ_DET.PROPOSAL_METHOD = 'gt'

# Use RPN to detect objects
__C.TRAIN.OBJ_DET.HAS_RPN = True
# IOU >= thresh: positive example
__C.TRAIN.OBJ_DET.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
__C.TRAIN.OBJ_DET.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.OBJ_DET.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
__C.TRAIN.OBJ_DET.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.TRAIN.OBJ_DET.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
__C.TRAIN.OBJ_DET.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.OBJ_DET.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.OBJ_DET.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TRAIN.OBJ_DET.RPN_MIN_SIZE = 16
# Deprecated (outside weights)
__C.TRAIN.OBJ_DET.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.OBJ_DET.RPN_POSITIVE_WEIGHT = -1.0


#
# Testing options
#

__C.TEST = edict()

# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

#
# Testing options (object detection)
#

__C.TEST.OBJ_DET = edict()

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.OBJ_DET.NMS = 0.3

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.OBJ_DET.SVM = False

# Test using bounding-box regressors
__C.TEST.OBJ_DET.BBOX_REG = True

# Propose boxes
__C.TEST.OBJ_DET.HAS_RPN = False

# Test using these proposals
__C.TEST.OBJ_DET.PROPOSAL_METHOD = 'gt'

## NMS threshold used on RPN proposals
__C.TEST.OBJ_DET.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.OBJ_DET.RPN_PRE_NMS_TOP_N = 6000
## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.OBJ_DET.RPN_POST_NMS_TOP_N = 300
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.OBJ_DET.RPN_MIN_SIZE = 16


#
# MISC
#

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Shuffle directory
__C.SHUFFLE_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'output','shuffled_sets'))

# Model directory
__C.MODELS_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'models', 'coco'))

# Place outputs under an experiments directory
__C.EXP_DIR = "default"

# Default GPU device id
__C.GPU_ID = 0

# path to save information for imdb report
__C.IMDB_REPORT_OUTPUT_PATH = "output/imdbReport/"

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.OBJ_DET = edict()

__C.OBJ_DET.DEDUP_BOXES = 1./16.

# Use GPU implementation of non-maximum suppression
__C.OBJ_DET.USE_GPU_NMS = True

# How much information about the bounding boxes do we store in memory?
__C.OBJ_DET.BBOX_VERBOSE = True


def get_output_dir(imdb, net=None):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if net is not None:
        outdir = osp.join(outdir, net.name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        old_type = type(b[k])
        # the types must match, too; unless old_type is not edict and not None
        if old_type is not type(v) and (old_type is edict and old_type is not type(None)):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        elif v == "None":
            b[k] = None
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfgData_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C.DATASETS)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value

def createPathSetID(setID):
    return osp.join(cfg.PATH_MIXTURE_DATASETS,setID)

def createPathRepeat(setID,r):
    return osp.join(cfg.PATH_MIXTURE_DATASETS,setID,r)
    
def createFilenameID(setID,r,size):
    return osp.join(cfg.PATH_MIXTURE_DATASETS,setID,r,size)



