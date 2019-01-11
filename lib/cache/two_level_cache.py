import os,uuid
import os.path as osp
from easydict import EasyDict as edict
from utils.base import readPickle,writePickle
from cache.one_level_cache import Cache

class TwoLevelCache(object):

    def __init__(self,cache_dir,cacheConfig,settings):
        self.cache_dir = cache_dir
        self.settings = settings
        self.cacheConfig = cacheConfig
        lookup_filename = osp.join(self.cache_dir,'lookup.pkl')
        self.lookup_cache = Cache(lookup_filename,self.cacheConfig)
    
    def save(self,payload):
        uuid_str = str(uuid.uuid4())
        self.payload_filename = osp.join(self.cache_dir,'{}.pkl'.format(uuid_str))
        if osp.exists(self.cache_dir) is False:
            os.makedirs(self.cache_dir)
        writePickle(self.payload_filename,payload)
        self.lookup_cache.save(self.payload_filename)

    def load_payload_filename(self):
        # 1. collect all the matches in a list
        ds_filenames = self.lookup_cache.load_all_matches()
        if len(ds_filenames) is 0:
            return None
        # 2. use policy to determine which matches to use
        if self.settings is None:
            match_index = 0
        else:
            match_index = self.settings.match_index
        return ds_filenames[match_index]

    def load(self):
        self.payload_filename = self.load_payload_filename()
        print(self.payload_filename)
        if self.payload_filename:
            payload = readPickle(self.payload_filename)
            return payload
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

# payloadCacheCfg.CACHE_PROMPT = edict()
# payloadCacheCfg.CACHE_PROMPT.DATASET_AUGMENTATION.RANDOMIZE = True

# payloadCacheCfg.CACHE_PROMPT.DATASET_AUGMENTATION.RANDOMIZE.BOOL = True
# payloadCacheCfg.CACHE_PROMPT.DATASET_AUGMENTATION.RANDOMIZE.RESAMPLE = True

# payloadCacheCfg.CACHE_PROMPT.DATASET_AUGMENTATION.RANDOMIZE_SUBSET.BOOL = True
# payloadCacheCfg.CACHE_PROMPT.DATASET_AUGMENTATION.RANDOMIZE_SUBSET.RESAMPLE = True

# payloadCacheCfg.CACHE_PROMPT.DATASET.SUBSAMPLE.BOOL = True
# payloadCacheCfg.CACHE_PROMPT.DATASET.SUBSAMPLE.RESAMPLE = True



