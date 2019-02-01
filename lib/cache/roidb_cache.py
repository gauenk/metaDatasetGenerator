import os,uuid
import os.path as osp
from easydict import EasyDict as edict

from utils.base import readPickle,writePickle
from cache.two_level_cache import TwoLevelCache

"""
roidbCacheCfg = edict() # set of parameters to follow when loading the dataset from the lookup cache
roidbCacheCfg.CACHE_PROMPT = edict()
roidbCacheCfg.CACHE_PROMPT.DATASET_AUGMENTATION_RANDOMIZE = True
roidbCacheCfg.CACHE_PROMPT.DATASET_AUGMENTATION_RANDOMIZE_SUBSET = True
roidbCacheCfg.CACHE_PROMPT.DATASET.SUBSAMPLE_BOOL = True
"""

class RoidbCache(TwoLevelCache):

    def __init__(self,cache_path,imdb_str,cfg,ds_config,roidb_settings,lookup_id,fieldname):
        root_dir = osp.join(cache_path,imdb_str)
        self.cacheConfig = self.construct_data_cache_config(cfg,ds_config)
        super(RoidbCache,self).__init__(root_dir,self.cacheConfig,roidb_settings,lookup_id,fieldname)

    def construct_data_cache_config(self,cfg,ds_config):
        cacheDataCfg = edict()

        cacheDataCfg.task = cfg.TASK
        cacheDataCfg.subtask = cfg.SUBTASK

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

    def save(self,fieldname,*args,**kwargs):
        self.lookup_cache.fieldname = fieldname
        super(RoidbCache,self).save(*args,**kwargs)

    def load(self,fieldname,*args,**kwargs):
        self.lookup_cache.fieldname = fieldname
        return super(RoidbCache,self).load(*args,**kwargs)
        
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



