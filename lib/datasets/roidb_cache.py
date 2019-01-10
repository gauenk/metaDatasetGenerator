import os,uuid
import os.path as osp
from easydict import EasyDict as edict

from utils.base import readPickle,writePickle
from utils.cache import Cache

"""
roidbCacheCfg = edict() # set of parameters to follow when loading the dataset from the lookup cache
roidbCacheCfg.CACHE_PROMPT = edict()
roidbCacheCfg.CACHE_PROMPT.DATASET_AUGMENTATION_RANDOMIZE = True
roidbCacheCfg.CACHE_PROMPT.DATASET_AUGMENTATION_RANDOMIZE_SUBSET = True
roidbCacheCfg.CACHE_PROMPT.DATASET.SUBSAMPLE_BOOL = True
"""

class RoidbCache():

    def __init__(self,root_dir,imdb_str,cfg,ds_config,roidb_cache_cfg):
        self.cache_dir = osp.join(root_dir,imdb_str)
        self.config = roidb_cache_cfg
        self.cacheConfig = self.construct_data_cache_config(cfg,ds_config)
        lookup_filename = osp.join(self.cache_dir,'lookup.pkl')
        self.lookup_cache = Cache(lookup_filename,self.cacheConfig)

    def construct_data_cache_config(self,cfg,ds_config):
        cacheDataCfg = edict()

        cacheDataCfg.task = cfg.TASK
        cacheDataCfg.subtask = cfg.SUBTASK

        # why should "dataset_augmentation" be in the cache at all? do we save something associated with the dataset augmentation? I don't think I do currently...
        cacheDataCfg.dataset_augmentation = edict() # primary config set 1
        cacheDataCfg.dataset_augmentation.bool_value = cfg.DATASET_AUGMENTATION.BOOL
        #cacheDataCfg.dataset_augmentation.configs = cfg.DATASET_AUGMENTATION.CONFIGS # says: "what type of augmentations do we have?"
        cacheDataCfg.dataset_augmentation.image_translate = cfg.DATASET_AUGMENTATION.IMAGE_TRANSLATE
        cacheDataCfg.dataset_augmentation.image_rotate = cfg.DATASET_AUGMENTATION.IMAGE_ROTATE
        cacheDataCfg.dataset_augmentation.image_crop = cfg.DATASET_AUGMENTATION.IMAGE_CROP
        cacheDataCfg.dataset_augmentation.percent_augmentations_used = cfg.DATASET_AUGMENTATION.N_PERC # says: "how many of the possible augmentations should we use for each augmented sample?"

        cacheDataCfg.dataset_augmentation.percent_samples_augmented = cfg.DATASET_AUGMENTATION.N_SAMPLES # says: "how many samples are we augmenting?"
        # cacheDataCfg.dataset_augmentation.bool_by_samples = cfg.DATASET_AUGMENTATION.SAMPLE_BOOL_VECTOR # says: which samples are augmented?
        cacheDataCfg.dataset_augmentation.randomize = cfg.DATASET_AUGMENTATION.RANDOMIZE
        cacheDataCfg.dataset_augmentation.randomize_subset = cfg.DATASET_AUGMENTATION.RANDOMIZE_SUBSET

        cacheDataCfg.dataset = edict() # primary config set 2
        cacheDataCfg.dataset.subsample_bool = cfg.DATASETS.SUBSAMPLE_BOOL
        cacheDataCfg.dataset.annotation_class = cfg.DATASETS.ANNOTATION_CLASS
        # cacheDataCfg.dataset.size = cfg.DATASETS.SIZE #len(roidb) ## We can't use this because it is set by the unfiltered image_index variable
        # cacheDataCfg.dataset.classes = cfg.DATASETS.CLASSES #imdb.classes; classes should already be filtered if they need to be ## can't use this it is unfiltered

        cacheDataCfg.filters = edict() # primary config set 3
        cacheDataCfg.filters.classes = cfg.DATASETS.FILTERS.CLASS
        cacheDataCfg.filters.empty_annotations = cfg.DATASETS.FILTERS.EMPTY_ANNOTATIONS

        cacheDataCfg.misc = edict() # primary config set 3
        cacheDataCfg.misc.use_diff = ds_config['use_diff']
        cacheDataCfg.misc.rpn_file = ds_config['rpn_file']
        cacheDataCfg.misc.min_size = ds_config['min_size']
        cacheDataCfg.misc.flatten_image_index = ds_config['flatten_image_index']
        cacheDataCfg.misc.setID = ds_config['setID']

        return cacheDataCfg
        # TODO: put theses somewhere else
        # ? assert len(roidb) == cacheDataCfg.DATASET.SIZE
        # ? assert imdb.classes == cacheDataCfg.DATASET.CLASSES
    
    def save(self,roidb):
        # cache the roidb
        uuid_str = str(uuid.uuid4())
        self.roidb_filename = osp.join(self.cache_dir,'{}.pkl'.format(uuid_str))
        if osp.exists(self.cache_dir) is False:
            os.makedirs(self.cache_dir)
        writePickle(self.roidb_filename,roidb)
        self.lookup_cache.save(self.roidb_filename)

    def load_roidb_filename(self):
        # 1. collect all the matches in a list
        ds_filenames = self.lookup_cache.load_all_matches()
        if len(ds_filenames) is 0:
            return None
        # 2. use policy to determine which matches to use
        if self.config is None:
            match_index = 0
        else:
            match_index = self.config.match_index
        return ds_filenames[match_index]

    def load(self):
        self.roidb_filename = self.load_roidb_filename()
        print(self.roidb_filename)
        if self.roidb_filename:
            roidb = readPickle(self.roidb_filename)
            return roidb
        else:
            return None
    
    def print_dataset_summary_by_uuid(datasetConfigList,uuid_str):
        pass
    
# TODO: fix the bottom for a prompt to resample the relatd values
# if self.config.cache_prompt_bool is False:
#     return ds_filenames[match_index]
# else:
#     # run the cache_prompt
#     for argument in self.config.cache_prompt_list:
#         if self.configToCompare[argument] is True:
#             match_index = promptUserForMatchIndex(argument,match_list)
#             if match_index == -1:
#                 return None

# roidbCacheCfg.CACHE_PROMPT = edict()
# roidbCacheCfg.CACHE_PROMPT.DATASET_AUGMENTATION.RANDOMIZE = True

# roidbCacheCfg.CACHE_PROMPT.DATASET_AUGMENTATION.RANDOMIZE.BOOL = True
# roidbCacheCfg.CACHE_PROMPT.DATASET_AUGMENTATION.RANDOMIZE.RESAMPLE = True

# roidbCacheCfg.CACHE_PROMPT.DATASET_AUGMENTATION.RANDOMIZE_SUBSET.BOOL = True
# roidbCacheCfg.CACHE_PROMPT.DATASET_AUGMENTATION.RANDOMIZE_SUBSET.RESAMPLE = True

# roidbCacheCfg.CACHE_PROMPT.DATASET.SUBSAMPLE.BOOL = True
# roidbCacheCfg.CACHE_PROMPT.DATASET.SUBSAMPLE.RESAMPLE = True



