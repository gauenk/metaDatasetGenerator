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

import os,pickle,uuid,copy
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

__C.FRAMEWORK = 'caffe'
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
cfg.MODEL_NAME_APPEND_STRING = None

#
# Global Input Options
#

__C.INPUT_DATA = edict()
__C.INPUT_DATA.BIJECTION = None

__C.WARP_AFFINE = edict()
__C.WARP_AFFINE.BOOL = True
__C.WARP_AFFINE.PRETRAIN = False

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
__C.TRAIN.SNAPSHOT_ITERS = 3000

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


__C.TEST.INPUTS = edict()
__C.TEST.INPUTS.IM_INFO = False
__C.TEST.INPUTS.RESHAPE = True

__C.TEST.CREATE_ANGLE_DATASET = False


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
__C.CROPPED_IMAGE_SIZE = 32

# The size of the input for raw images
__C.RAW_IMAGE_SIZE = 300

# The size of the input for raw images
__C.AL_IMAGE_SIZE = 28

# The size of the input for activations
__C.AV_IMAGE_SIZE = 400
__C.AV_COLOR_CHANNEL = 4

# actual image size for input data
cfg.IMAGE_SIZE = [cfg.CROPPED_IMAGE_SIZE,cfg.CROPPED_IMAGE_SIZE]
cfg.SIAMESE_IMAGE_SIZE = cfg.IMAGE_SIZE

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

def saveExperimentConfig():
    import yaml
    uuid_str = str(uuid.uuid4())
    filename = "./output/experiment_cfgs/{}.yml".format(uuid_str)
    with open(filename, 'w') as f:
        yaml_str = yaml.dump(cfg)
        f.write(yaml_str)
    print("saved experiment config to {}".format(filename))

def GET_SAVE_ACTIVITY_VECTOR_BLOBS_DIR(exp_dir=None):
    if exp_dir is None:
        dirn = "./output/activations/{}/{}-{}-{}/".format(__C.EXP_DIR.replace("/",""),cfg.DATASETS.CALLING_DATASET_NAME,cfg.DATASETS.CALLING_IMAGESET_NAME,cfg.DATASETS.CALLING_CONFIG)
    else:
        dirn = "./output/activations/{}/{}-{}-{}/".format(exp_dir,cfg.DATASETS.CALLING_DATASET_NAME,cfg.DATASETS.CALLING_IMAGESET_NAME,cfg.DATASETS.CALLING_CONFIG)
    if not osp.exists(dirn):
        os.makedirs(dirn)
    return dirn

# used for saving activity vectors
__C.ACTIVATIONS = edict()
__C.ACTIVATIONS.SAVE_BOOL = False
__C.ACTIVATIONS.LAYER_NAMES = ['conv1','conv2','ip1','cls_score','cls_prob']
#__C.ACTIVATIONS.LAYER_NAMES = ['warp_angle']
#__C.ACTIVATIONS.LAYER_NAMES = []
__C.ACTIVATIONS.GET_SAVE_DIR = GET_SAVE_ACTIVITY_VECTOR_BLOBS_DIR
__C.ACTIVATIONS.SAVE_OBJ = 'order' # 'image_id','order'
#__C.ACTIVATIONS.INPUT_SIZE = [84,84,3] # not need to only have 3 channels, but we can for now
__C.ACTIVATIONS.CACHE_ID = None #for agg_activation cache; each layer gets its own id
__C.CACHE.DATA.ACTIVATION = None


# __C.GET_SAVE_ACTIVITY_VECTOR_BLOBS_DIR = GET_SAVE_ACTIVITY_VECTOR_BLOBS_DIR
# # __C.SAVE_ACTIVITY_VECTOR_BLOBS = [] # the list of blobs to save
# __C.SAVE_ACTIVITY_VECTOR_BLOBS = __C.ACTIVATION_VALUES.LAYER_NAMES #['conv1','conv2','ip1','cls_score','cls_prob']
# __C.SAVE_ACTIVITY_VECTOR_BLOBS_WITH_KEYS = False

__C.TP_FN_RECORDS_WITH_KEYS = False
#__C.ACTIVATION_VALUES.FOR_CACHE = [cfg.ACTIVATION_VALUES.LAYER_NAMES,cfg.ACTIVATION_VALUES.SAVE_DIR,cfg.ACTIVATION_VALUES.SAVE_OBJ,cfg.SAVE_ACTIVITY_VECTOR_BLOBS,cfg.SAVE_ACTIVITY_VECTOR_BLOBS_WITH_KEYS]
# Active Learning Settings
__C.ACTIVE_LEARNING = edict()
__C.ACTIVE_LEARNING.N_ITERS = 11000
__C.ACTIVE_LEARNING.VAL_SIZE = 30000
__C.ACTIVE_LEARNING.SUBSET_SIZE = 500
__C.ACTIVE_LEARNING.N_COVERS = 300
__C.ACTIVE_LEARNING.REPORT = False
__C.ACTIVE_LEARNING.LAYER_NAMES = ['conv1','conv2','ip1','cls_score','cls_prob']
__C.ACTIVE_LEARNING.FOR_CACHE = [cfg.ACTIVE_LEARNING.N_ITERS,cfg.ACTIVE_LEARNING.VAL_SIZE,cfg.ACTIVE_LEARNING.SUBSET_SIZE,cfg.ACTIVE_LEARNING.N_COVERS,cfg.ACTIVE_LEARNING.LAYER_NAMES]

# Prune Network: the number is the modulous for the frequency of pruning
cfg.PRUNE_NET = 0 

# add noise to image inputs
__C.TRAIN.IMAGE_NOISE = False
__C.TEST.IMAGE_NOISE = False
__C.IMAGE_NOISE = False

__C.TRAIN.MAX_ITERS = None

#
# applying a single random rotation to each input sample...
#

# NOTE: the scope of this being a global setting has troubling implications .... it should be more local to the DataLoader Object...

cfg.TRANSFORM_EACH_SAMPLE = edict()

DATA_0 = edict()
DATA_0.BOOL = False
DATA_0.RAND = True
DATA_0.TYPE = "rotate"
DATA_0.PARAMS = {'angle_min':0.,'angle_max':360.}

# order of the list is the order the transformations are applied
cfg.TRANSFORM_EACH_SAMPLE.DATA_LIST = [DATA_0]

LABEL_0 = edict()
LABEL_0.BOOL = False
LABEL_0.RAND = True
LABEL_0.TYPE = "file_replace"
LABEL_0.PARAMS = {'filename':None,'labels':None,'index_type':'roidb_index'}

LABEL_1 = edict()
LABEL_1.BOOL = False
LABEL_1.TYPE = "angle"
LABEL_1.PARAMS = {"angle_index":2}

# order of the list is the order the transformations are applied
cfg.TRANSFORM_EACH_SAMPLE.LABEL_LIST = [LABEL_0,LABEL_1]

#
# Addition input from the original sample
#

cfg.ADDITIONAL_INPUT = edict()
cfg.ADDITIONAL_INPUT.BOOL = False
cfg.ADDITIONAL_INPUT.EXP_CFG_FILE = ""
cfg.ADDITIONAL_INPUT.TYPE = None #'activations' #'image'
cfg.ADDITIONAL_INPUT.ACTIVATIONS = edict()
cfg.ADDITIONAL_INPUT.ACTIVATIONS.LAYER_NAMES = ['conv1','conv2','ip1','cls_score','cls_prob']
cfg.ADDITIONAL_INPUT.ACTIVATIONS.INPUT_SIZE = [84,84,3]
cfg.ADDITIONAL_INPUT.ACTIVATIONS.GET_SAVE_DIR = GET_SAVE_ACTIVITY_VECTOR_BLOBS_DIR
cfg.ADDITIONAL_INPUT.ACTIVATIONS.SAVE_OBJ = 'order'
cfg.ADDITIONAL_INPUT.ACTIVATIONS.SAVE_BOOL = False
cfg.ADDITIONAL_INPUT.INFO = {'image':
                             {'axis':1},
                             'activations':
                             {"size_by_layer":{},
                              'net_name':""}
                             }

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

def setModelInfo(solverPrototxt,solverInfo=None):
    if solverInfo is None:
        solverInfo = solverPrototxtInfoDict(solverPrototxt)
    setModelInfoFromSolverInfo(solverInfo)

def setModelInfoFromSolverInfo(solverInfo):
    cfg.modelInfo.imdb_str = cfg.DATASETS.CALLING_IMDB_STR
    cfg.modelInfo.architecture = solverInfo['arch']
    cfg.modelInfo.optim = solverInfo['optim'].lower()

    if 'infix_str' in solverInfo.keys():
        cfg.modelInfo.infix_str = solverInfo['infix_str']
    else:
        cfg.modelInfo.infix_str = False

    if 'siamese' in solverInfo.keys():
        cfg.modelInfo.siamese = solverInfo['siamese']
    else:
        cfg.modelInfo.siamese = False        

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
        cfg.modelInfo.class_filter = False
    else:
        cfg.modelInfo.class_filter = len(cfg.DATASETS.FILTERS.CLASS_INCLUSION_LIST)

    if cfg.WARP_AFFINE.BOOL is True and cfg.WARP_AFFINE.PRETRAIN is True:
        cfg.modelInfo.warp_affine_pretrain = True
        cfg.modelInfo.warp_affine_pretrain_net_name = cfg.WARP_AFFINE.PRETRAIN_NAME
    else:
        cfg.modelInfo.warp_affine_pretrain = False

    if cfg.TEST_NET.NET_PATH is not None:
        cfg.modelInfo.loaded_model = os.path.basename(cfg.TEST_NET.NET_PATH)


    cfg.modelInfo.additional_input = edict()
    cfg.modelInfo.additional_input.bool = False
    setAdditionInput(cfg.modelInfo)
    # if cfg.ADDITIONAL_INPUT.BOOL:
    #     # input sample with:
    #     # (i) activations; (ii) another image


    create_snapshot_prefix(cfg.modelInfo)

# functions for filling in a prototxt with other prototxts for training.... should probably swtich to pytorch soon...

def check_config_for_error():
    # handle region proposal network
    if cfg.TEST.OBJ_DET.HAS_RPN is False and cfg.TASK == 'object_detection':
        raise ValueError("We can't handle rpn correctly. See [box_proposals] in original faster-rcnn code.")

def setAdditionInput(modelInfo):

    #TODO send in agg activation value model and ensure the data_loader gets that info.
    # if cfg.ADDITIONAL_INPUT.BOOL and cfg.ADDITIONAL_INPUT.TYPE == "activation":
    #     if cfg.ADDITIONAL_INPUT.ACTIVATION_MODEL is None or \
    #        cfg.ADDITIONAL_INPUT.ACTIVATION_IMDB is None:
    #         print("ERROR: we need a model to load activations from")

    modelInfo.additional_input.bool = cfg.ADDITIONAL_INPUT.BOOL
    modelInfo.additional_input.type = cfg.ADDITIONAL_INPUT.TYPE
    modelInfo.additional_input.info = cfg.ADDITIONAL_INPUT.INFO
    if modelInfo.additional_input.type == 'activations':
        settings = modelInfo.additional_input.info['activations']
        settings.agg = None
        settings.cfg = readYamlToEdict(cfg.ADDITIONAL_INPUT.EXP_CFG_FILE)
        settings.activations_cfg = cfg.ADDITIONAL_INPUT.ACTIVATIONS
        settings['size_by_layer'] = {layerName:np.arange(50176) for layerName in settings.activations_cfg.LAYER_NAMES}
        if len(settings.activations_cfg.LAYER_NAMES) == 0:
            print("[config] addition_input type uses activations but none loaded. quitting.")
            exit()

def computeUpdatedConfigInformation():
    # compute the input image size
    cfg.IMAGE_SIZE = [cfg.CROPPED_IMAGE_SIZE,cfg.CROPPED_IMAGE_SIZE]
    if cfg.COLOR_CHANNEL == 1:
        cfg.PIXEL_MEANS = np.array([np.mean(cfg.PIXEL_MEANS)])
    if cfg.ADDITIONAL_INPUT.BOOL:
        if cfg.ADDITIONAL_INPUT.TYPE == 'image' and cfg.modelInfo.siamese is False:
            axis = cfg.ADDITIONAL_INPUT.INFO['image']['axis']
            #cfg.IMAGE_SIZE[axis] *= 2
        if cfg.ADDITIONAL_INPUT.TYPE == 'activations' and cfg.modelInfo.siamese is True:
            cfg.SIAMESE_IMAGE_SIZE = cfg.ADDITIONAL_INPUT.ACTIVATIONS.INPUT_SIZE

# TODO: what do we actually need for a data cache?
