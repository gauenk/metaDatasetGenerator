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

import os,pickle
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
from utils.bijectionFunctions import BijectionFunctions
from core.configBase import *
from core.configData import *
from core.configDatasetAugmentation import *

#
# Dataset options
#
__C = cfg

#
# Global Options
#

__C.BATCH_SIZE = None
__C.SCALES = None

__C.AL_CLS = edict()
__C.AL_CLS.BALANCE_CLASSES = True
__C.AL_CLS.LAYERS = ['conv5_1','conv4_1','conv3_1']
__C.AL_CLS.ENTROPY_SUMMARY = False

__C.TEST_NET = edict()
__C.TEST_NET.NET_PATH = ""
__C.TEST_NET.DEF_PATH = ""


cfg.modelInfo = edict()

#
# Global Input Options
#

__C.INPUT_DATA = edict()
__C.INPUT_DATA.BIJECTION = None

#
# Training options
#

__C.TRAIN = edict()
#__C.PATH_YMLDATASETS = "helps"
__C.PATH_YMLDATASETS = "gauenk"
__C.PATH_MIXTURE_DATASETS = "./data/mixtureDatasets/"

# create new snapshot name?
cfg.TRAIN.RECREATE_SNAPSHOT_NAME = True

# limit the number of annotations in a dataset
cfg.TRAIN.CLIP_SIZE = None

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)
#__C.TRAIN.SCALES = (64,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000
#__C.TRAIN.MAX_SIZE = 64

# Images to use per minibatch
__C.TRAIN.BATCH_SIZE = 1

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = False

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 5000

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

# Batch size
__C.TRAIN.BATCH_SIZE = 1

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

# VAE Options

__C.TRAIN.VAE = edict()
__C.TRAIN.VAE.BATCH_SIZE = __C.TRAIN.BATCH_SIZE

__C.TRAIN.CLASSIFICATION = edict()
__C.TRAIN.CLASSIFICATION.TASK = 'tp_fn'
__C.TRAIN.CLASSIFICATION.THRESHOLD = 0.5
__C.TRAIN.CLASSIFICATION.PROPOSAL_METHOD = 'gt'


# CLASSIFICATION Options

__C.TRAIN.CLS = edict()
__C.TRAIN.CLS.BATCH_SIZE = __C.TRAIN.BATCH_SIZE
__C.TRAIN.CLS.BALANCE_CLASSES = True

# AL Train Options

__C.TRAIN.AL_CLS = edict()
__C.TRAIN.AL_CLS.BALANCE_CLASSES = True
#__C.TRAIN.AL_CLS.LAYERS = ['conv5_1','conv4_3','conv4_1','conv3_3']
#__C.TRAIN.AL_CLS.LAYERS = ['conv5_1','conv4_3']
__C.TRAIN.AL_CLS.LAYERS = ['conv1','conv2','ip1','cls_score']


#
# Testing options
#

__C.TEST = edict()

# number of image per batch
__C.TEST.BATCH_SIZE = 1

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

# max # of detections per image over all classes
__C.TEST.OBJ_DET.MAX_PER_IMAGE = 100

__C.TEST.CLASSIFICATION = edict()
__C.TEST.CLASSIFICATION.TASK = 'tp_fn'
__C.TEST.CLASSIFICATION.THRESHOLD = 0.5
__C.TEST.CLASSIFICATION.PROPOSAL_METHOD = 'gt'

#
# AL Testing Options
#

__C.TEST.AL_CLS = edict()
__C.TEST.AL_CLS.BALANCE_CLASSES = True
__C.TEST.AL_CLS.LAYERS =  ['conv1','conv2','ip1','cls_score']


#
# MISC
#

# official names for publication
__C.DATASET_NAMES_PAPER = ['COCO', 'ImageNet', 'VOC', 'Caltech', 'INRIA', 'SUN', 'KITTI', 'CAM2','MNIST']
__C.DATASET_NAMES_ORDERED = ['coco', 'imagenet', 'pascal_voc', 'caltech', 'inria', 'sun','kitti','cam2','mnist','cifar_10']

# For print statements
__C.DEBUG = False
__C._DEBUG = edict()
__C._DEBUG.utils = edict()
__C._DEBUG.utils.misc = False
__C._DEBUG.core = edict()
__C._DEBUG.core.config = False
__C._DEBUG.core.test = False
__C._DEBUG.datasets = edict()
__C._DEBUG.datasets.repo_imdb = False
__C._DEBUG.datasets.evaluators = edict()
__C._DEBUG.datasets.evaluators.bbox_utils = False
__C._DEBUG.datasets.evaluators.bboxEvaluator = False
__C._DEBUG.datasets.repo_imdb = False
__C._DEBUG.rpn = edict()
__C._DEBUG.rpn.proposal_layer = False


# pixel mean values (BGR order) as a (1, 1, 3) array
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

# is it ssd?
__C.SSD = False

# input size of an ssd image
__C.SSD_img_size= 300

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.OBJ_DET = edict()

__C.OBJ_DET.DEDUP_BOXES = 1./16.

# Use GPU implementation of non-maximum suppression
__C.OBJ_DET.USE_GPU_NMS = True


# Threshold for the generative model
__C.GENERATE = edict()
__C.GENERATE.THRESHOLD = 127.5

# How much information about the bounding boxes do we store in memory?
__C.OBJ_DET.BBOX_VERBOSE = True

# The sizes used for creating the mixture datasets
#__C.MIXED_DATASET_SIZES = [10,1000,5000]
__C.MIXED_DATASET_SIZES = [50,1000,5000]

# The size of the input for images cropped to their annotations
__C.CROPPED_IMAGE_SIZE = 227

# The size of the input for raw images
__C.RAW_IMAGE_SIZE = 300

# The size of the input for raw images
__C.AL_IMAGE_SIZE = 28

# The size of the input for activations
__C.AV_IMAGE_SIZE = 400
__C.AV_COLOR_CHANNEL = 4

# the size of the 
__C.CONFIG_DATASET_INDEX_DICTIONARY_PATH = "default_dataset_index.yml"

# path to save information for imdb report
__C.IMDB_REPORT_OUTPUT_PATH = "output/imdbReport/"

# name that dataset! output
__C.PATH_TO_NTD_OUTPUT = "./output/ntd/"

# output for annotation analysis
__C.PATH_TO_ANNO_ANALYSIS_OUTPUT = "./output/annoAnalysis/"

# output for cross dataset generalization
__C.PATH_TO_X_DATASET_GEN = "./output/xDatasetGen/"

# bool for computing image statistics in rdl_load
__C.COMPUTE_IMG_STATS = True

# how much should we rotate each image?
__C.IMAGE_ROTATE = 0

# how much should we rotate each image?
__C.COLOR_CHANNEL = 3 #color == 3 | black&white == 1

# should we write the results?
__C.WRITE_RESULTS = True

# output for recoding the TP and FN of a model
__C.ROTATE_PATH = "./output/rotate/"

# a switching condition for different goals in training/testing
__C.TASK = "object_detection"
__C.SUBTASK = "default"

# string name of the output layer's probability vectore
__C.CLS_PROBS = "cls_prob"

# how to loadImage during testing
__C.LOAD_METHOD = None

def GET_SAVE_ACTIVITY_VECTOR_BLOBS_DIR():
    dirn = "./output/activity_vectors/{}/{}-{}-{}/".format(__C.EXP_DIR.replace("/",""),cfg.DATASETS.CALLING_DATASET_NAME,cfg.DATASETS.CALLING_IMAGESET_NAME,cfg.DATASETS.CALLING_CONFIG)
    if not osp.exists(dirn):
        os.makedirs(dirn)
    return dirn

# used for saving activity vectors
__C.ACTIVATION_VALUES = edict()
__C.ACTIVATION_VALUES.SAVE_BOOL = False
__C.ACTIVATION_VALUES.LAYER_NAMES = ['conv1','conv2','ip1','cls_score','cls_prob']
__C.ACTIVATION_VALUES.SAVE_DIR = GET_SAVE_ACTIVITY_VECTOR_BLOBS_DIR
__C.ACTIVATION_VALUES.SAVE_OBJ = 'order' # 'image_id','order'
__C.GET_SAVE_ACTIVITY_VECTOR_BLOBS_DIR = GET_SAVE_ACTIVITY_VECTOR_BLOBS_DIR
# __C.SAVE_ACTIVITY_VECTOR_BLOBS = [] # the list of blobs to save
__C.SAVE_ACTIVITY_VECTOR_BLOBS = ['conv1','conv2','ip1','cls_score','cls_prob']
__C.SAVE_ACTIVITY_VECTOR_BLOBS_WITH_KEYS = False
__C.TP_FN_RECORDS_WITH_KEYS = False

# Active Learning Settings
__C.ACTIVE_LEARNING = edict()
__C.ACTIVE_LEARNING.N_ITERS = 11000
__C.ACTIVE_LEARNING.VAL_SIZE = 30000
__C.ACTIVE_LEARNING.SUBSET_SIZE = 500
__C.ACTIVE_LEARNING.N_COVERS = 300
__C.ACTIVE_LEARNING.REPORT = False
__C.ACTIVE_LEARNING.LAYER_NAMES = ['conv1','conv2','ip1','cls_score','cls_prob']

# Prune Network: the number is the modulous for the frequency of pruning
cfg.PRUNE_NET = 0 

# add noise to image inputs
__C.TRAIN.IMAGE_NOISE = False
__C.TEST.IMAGE_NOISE = False
__C.IMAGE_NOISE = False


# __C.TRAIN.DATASET_AUGMENTATION = edict()
# __C.TRAIN.DATASET_AUGMENTATION.SIZE = 64 # TODO: set dynamically later
# __C.TRAIN.DATASET_AUGMENTATION.EXHAUSTIVE = True
# __C.TRAIN.DATASET_AUGMENTATION.IMAGE_ROTATE_LIST=0
# __C.TRAIN.DATASET_AUGMENTATION.IMAGE_CROP_LIST=0
# __C.TRAIN.DATASET_AUGMENTATION.IMAGE_TRANSLATE_LIST=[[2,2,2,2]] # list of lists: [ [up,down,left,right], ...]; each list element (yes; the element itself is a list) is for *one* transformation


def get_output_dir(imdb_name, net=None):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    if type(imdb_name) is not str:
        imdb_name = imdb_name.name
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb_name))
    if net is not None:
        outdir = osp.join(outdir, net.name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def set_global_cfg(MODE):
    # set global variables for testing
    cfg.PIXEL_MEANS = np.array(cfg.PIXEL_MEANS)
    if MODE == "TEST":
        cfg.BATCH_SIZE = cfg.TEST.BATCH_SIZE
        cfg.AL_CLS.LAYERS = cfg.TEST.AL_CLS.LAYERS
        cfg.SCALES = cfg.TEST.SCALES
        cfg.IMAGE_NOISE = cfg.TEST.IMAGE_NOISE
    elif MODE == "TRAIN":
        cfg.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
        cfg.AL_CLS.LAYERS = cfg.TRAIN.AL_CLS.LAYERS
        cfg.SCALES = cfg.TRAIN.SCALES
        cfg.IMAGE_NOISE = cfg.TRAIN.IMAGE_NOISE
    maxPixelNum = int(max(__C.TRAIN.SCALES) * __C.TRAIN.MAX_SIZE / 4.)
    cfg.INPUT_DATA.BIJECTION = BijectionFunctions('rtVal = self.random_pixel_shuffle({})',maxPixelNum=maxPixelNum)

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

def iconicImagesFileFormat():
    if not osp.exists(cfg.PATH_TO_NTD_OUTPUT):
        os.makedirs(cfg.PATH_TO_NTD_OUTPUT)
    return osp.join(cfg.PATH_TO_NTD_OUTPUT,"{}")

def createPathSetID(setID):
    return osp.join(cfg.PATH_MIXTURE_DATASETS,setID)

def createPathRepeat(setID,r):
    return osp.join(cfg.PATH_MIXTURE_DATASETS,setID,r)
    
def createFilenameID(setID,r,size):
    return osp.join(cfg.PATH_MIXTURE_DATASETS,setID,r,size)

def loadDatasetIndexDict():
    # legacy
    return __C.DATASET_NAMES_ORDERED
    fn = osp.join(__C.ROOT_DIR,"./lib/datasets/ymlConfigs",cfg.CONFIG_DATASET_INDEX_DICTIONARY_PATH)
    import yaml
    with open(fn, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    indToCls = [None for _ in range(len(yaml_cfg))]
    for k,v in yaml_cfg.items():
        indToCls[v] = k
    while(None in indToCls):
        indToCls.remove(None)
    return indToCls

__C.clsToSet = __C.DATASET_NAMES_ORDERED

def getTestNetConfig(caffemodel,prototxt):
    # find model number of iterations 
    cfg.TEST_NET.NET_PATH = osp.splitext(caffemodel.split('/')[-1])[0]
    cfg.TEST_NET.DEF_PATH = osp.splitext(prototxt.split('/')[-1])[0]

def setModelInfo(solverPrototxt):
    solverInfo = solverPrototxtInfoDict(solverPrototxt)
    cfg.modelInfo.train_set = solverInfo['ds_name']
    cfg.modelInfo.architecture = solverInfo['arch']
    cfg.modelInfo.optim = solverInfo['optim'].lower()
    if cfg.TRAIN.IMAGE_NOISE is None or cfg.TRAIN.IMAGE_NOISE is 0:
        cfg.modelInfo.image_noise = False
    else:
        cfg.modelInfo.image_noise = cfg.TRAIN.IMAGE_NOISE

    if cfg.PRUNE_NET is None or cfg.PRUNE_NET is 0:
        cfg.modelInfo.prune = False
    else:
        cfg.modelInfo.prune = cfg.PRUNE_NET

    if cfg.DATASET_AUGMENTATION.BOOL:
        value = str(round(cfg.DATASET_AUGMENTATION.N_SAMPLES * 100,3))
        cfg.modelInfo.dataset_augmentation = value.replace('.','-')
    else:
        cfg.modelInfo.dataset_augmentation = False

    cls_incl_list_0_bool = len(cfg.DATASETS.FILTERS.CLASS_INCLUSION_LIST) == 0
    cls_filter_none_bool = cfg.DATASETS.FILTERS.CLASS is None
    cls_filter_false_bool = cfg.DATASETS.FILTERS.CLASS is False
    if cls_filter_none_bool or cls_filter_false_bool or cls_incl_list_0_bool:
        cfg.modelInfo.classFilter = False
    else:
        cfg.modelInfo.classFilter = len(cfg.DATASETS.FILTERS.CLASS_INCLUSION_LIST)

# functions for filling in a prototxt with other prototxts for training.... should probably swtich to pytorch soon...

def mangleFillmePrototxtTemplateFilename(prototxt_path):
    path = os.path.dirname(prototxt_path)
    prototxt = os.path.basename(prototxt_path)
    prototxt_base = prototxt.split('.')[0]
    if '_template' in prototxt_base:
        new_prototxt = prototxt_base.replace('_template','') + '.prototxt'
    else:
        new_prototxt = prototxt_base + 'fillme-in.prototxt'
    new_prototxt_path = os.path.join(path,new_prototxt)
    return new_prototxt_path

def findFillMeLayers(net):
    layer_list = [l for l in net.layer]
    found = []
    for layer in layer_list:
        layerName = layer.name
        print(layerName)
        if 'fillme' in layerName.lower():
            found.append(layer)
    return found

def getFillmeLayerFilename(fillmeLayer):
    # SO HACKY; only temporary solution until change to pytorch
    return fillmeLayer.convolution_param.weight_filler.type

def getLayerInfo(layer):
    if type(layer.bottom) is unicode:
        layer_bottom = str(layer.bottom)
    else:
        layer_bottom = [b for b in layer.bottom]
    if type(layer.top) is unicode:
        layer_top = str(layer.top)
    else:
        layer_top = [b for b in layer.top]
    layer_name = str(layer.name)
    # print(layer_name,layer_top)
    return layer_bottom,layer_top,layer_name

def replaceFillmeLayerValues(layer,fillmeNet):
    old_layer_bottom,old_layer_top,layer_name = getLayerInfo(layer)
    oglayer_name_base = layer_name.split('-')[0]
    replaceName = {}
    for l in fillmeNet.layer:
        l_name = l.name + '-' + oglayer_name_base
        l_bottoms = [b for b in l.bottom]
        l_tops = [b for b in l.bottom]
        replaceName[l.name] = l_name
        # print(l.name)
        l.name = l_name
        for idx,l_bottom in enumerate(l_bottoms):
            if l_bottom == 'bottom':
                l.bottom[idx] = old_layer_bottom
                replaceName[l_bottom] = old_layer_bottom
            elif l_bottom == 'bottom_0':
                l.bottom[idx] = old_layer_bottom[0]
                replaceName[l_bottom] = old_layer_bottom[0]
            elif l_bottom == 'bottom_1':
                l.bottom[idx] = old_layer_bottom[1]
                replaceName[l_bottom] = old_layer_bottom
        for idx,l_top in enumerate(l.top):
            if l_top == 'top':
                # print(l.top,old_layer_top)
                l.top[idx] = old_layer_top[idx]
                replaceName[l_top] = old_layer_top[idx]
            elif l_top == 'top_0':
                l.top[idx] = old_layer_top[0]
                replaceName[l_top] = old_layer_top[0]
            elif l_top == 'top_1':
                l.top[idx] = old_layer_top[1]
                replaceName[l_top] = old_layer_top
    # for replace strings now
    for l in fillmeNet.layer:
        # print(l.name)
        l_bottoms = [b for b in l.bottom]
        l_tops = [t for t in l.top]
        if l.name in replaceName.keys():
            l.name = replaceName[l.name]
        for idx,l_bottom in enumerate(l_bottoms):
            if l_bottom in replaceName.keys():
                l.bottom[idx] = replaceName[l_bottom]
        for idx,l_top in enumerate(l_tops):
            if l_top in replaceName.keys():
                if type(l.top) is unicode:
                    l.top = replaceName[l_top]
                else:
                    l.top[idx] = replaceName[l_top]

def replaceFillmeLayer(net,fillmeLayer):
    # get fillme net info
    fillmePrototxt = getFillmeLayerFilename(fillmeLayer)
    fillmeNet = buildNetFromPrototxt(fillmePrototxt)
    replaceFillmeLayerValues(fillmeLayer,fillmeNet)

    # use new net to replace current fillme layer
    """
    ** method outline **
    1. write layers before current layer to temporary file
    2. write the replacement network to temporary file
    3. write the remaining layers to the temporary file
    4. load file into network from temporary file
    5. remove temporary file
    """
    tmp_prototxt = 'tmp_file_prototxt.prototxt'
    for layer in net.layer:
        # print("written tmp_layer: {}".format(layer.name))
        if layer is fillmeLayer:
            appendPrototxtWithNet(fillmeNet,tmp_prototxt)
        else:
            appendPrototxtWithLayer(layer,tmp_prototxt)
    newNet = buildNetFromPrototxt(tmp_prototxt)
    os.remove(tmp_prototxt)
    return newNet

def replaceFillmeNetwork(prototxt):
    net = buildNetFromPrototxt(prototxt)
    fillMeLayerList = findFillMeLayers(net)
    while len(fillMeLayerList) > 0:
        layer = fillMeLayerList[0]
        print(layer.name)
        net = replaceFillmeLayer(net,layer)
        fillMeLayerList = findFillMeLayers(net)
    new_prototxt = mangleFillmePrototxtTemplateFilename(prototxt)
    writeNetFromProto(net,new_prototxt)


