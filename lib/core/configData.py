from core.configBase import *
from easydict import EasyDict as edict


cfg.DATASETS = edict()
cfgData = cfg.DATASETS
cfg.DATASETS.CALLING_DATASET_NAME = ""
cfg.DATASETS.CALLING_IMAGESET_NAME = ""
cfg.DATASETS.CALLING_CONFIG = ""
cfg.DATASETS.EXP_DATASET = ""
cfg.DATASETS.PATH_ROOT = ""
cfg.DATASETS.PATH_TO_IMAGES = ""
cfg.DATASETS.PATH_TO_ANNOTATIONS = ""
cfg.DATASETS.PATH_TO_IMAGESETS = ""
cfg.DATASETS.PATH_TO_RESULTS = ""
cfg.DATASETS.CLASSES = ""
cfg.DATASETS.SIZE = -1
cfg.DATASETS.COMPID = "defaultCompID"
cfg.DATASETS.IMAGE_TYPE = ""
cfg.DATASETS.ANNOTATION_TYPE = ""
cfg.DATASETS.PARSE_ANNOTATION_REGEX = None
cfg.DATASETS.CONVERT_TO_PERSON = None
cfg.DATASETS.IMAGE_INDEX_TO_IMAGE_PATH = None
cfg.DATASETS.USE_IMAGE_SET = None
cfg.DATASETS.CONVERT_ID_TO_CLS_FILE = None
cfg.DATASETS.ONLY_PERSON = False
cfg.DATASETS.MODEL = None
cfg.DATASETS.ANNOTATION_CLASS = "object_detection"
cfg.DATASETS.IS_IMAGE_INDEX_FLATTENED = False
cfg.DATASETS.HAS_BBOXES = False
cfg.DATASETS.SUBSAMPLE_SIZE = -1
cfg.DATASETS.CLASS_FILTER = True
cfg.DATASETS.CLASS_INCLUSION_LIST = []   # NOTE! THIS FEATURE IS INCOMPLETE!!
cfg.DATASETS.CLASS_INCLUSION_LIST_BY_DATASET = {
    'mnist': [3,8],
    'cifar_10': ['aeroplane','automobile']
}

def set_class_inclusion_list_by_calling_dataset():
    if cfg.DATASETS.CALLING_DATASET_NAME == '': raise ValueError("Dataset must be loaded before this function can be called")
    cfg.DATASETS.CLASS_INCLUSION_LIST = cfg.DATASETS.CLASS_INCLUSION_LIST_BY_DATASET[cfg.DATASETS.CALLING_DATASET_NAME]

def cfgData_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    merge_a_into_b(yaml_cfg, cfg.DATASETS)
    
    useClsDerived_Imagenet = (cfg.DATASETS.CALLING_DATASET_NAME == "imagenet_cls") and \
                             ("train" in cfg.DATASETS.CALLING_IMAGESET_NAME)
    if useClsDerived_Imagenet:
        cfgData['ANNOTATION_TYPE'] = "cls_derived"
    load_tp_fn_record_path()

