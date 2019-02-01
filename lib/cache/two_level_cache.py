import os,uuid,re
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
from utils.base import readPickle,writePickle,readNdarray,writeNdarray
from cache.one_level_cache import Cache

class TwoLevelCache(object):

    """
    one level of indirection for faster loading

    lookup cache format:
    [ unique_config_to_match_0 : unique_filename_0 ,
      unique_config_to_match_1 : unique_filename_1
                       .
                       .
                       .
    ]
    """


    def __init__(self,cache_dir,cacheConfig,settings,lookup_id,fieldname):
        self.cache_dir = cache_dir
        self.settings = settings
        self.cacheConfig = cacheConfig
        lookup_filename = 'lookup_{}.pkl'.format(lookup_id)
        lookup_path = osp.join(self.cache_dir,lookup_filename)
        self.lookup_cache = Cache(lookup_path,self.cacheConfig,fieldname)
    
    def update_config(self,new_config):
        """
        update "lookup" cache configuration
        """
        success = self.lookup_cache.update_config(new_config)
        if success:
            print("successfuly updated lookup cache config!")
        else:
            print("[two_level_cache] ERROR: no lookup cache to reset")
        
    
    def _get_save_filename(self,filenamePrefix,saveType):
        uuid_str = str(uuid.uuid4())
        ext_str = ''

        prefix_str = uuid_str
        if filenamePrefix is not None:
            prefix_str = filenamePrefix + "_" + uuid_str

        if saveType == "pickle":
            ext_str = 'pkl'
        elif saveType == "ndarray":
            ext_str = 'npy'
        else:
            print("[two_level_cache.py] Unknown savetype [{}]. quitting".format(saveType))
            exit()

        filename = osp.join(self.cache_dir,'{}.{}'.format(prefix_str,ext_str))
        return filename

    def save(self,payload,filenamePrefix=None,saveType="pickle"):
        self.payload_filename = self._get_save_filename(filenamePrefix,saveType)
        if osp.exists(self.cache_dir) is False:
            os.makedirs(self.cache_dir)
        if saveType == "pickle":
            writePickle(self.payload_filename,payload)
        elif saveType == "ndarray":
            writeNdarray(self.payload_filename,payload)
        else:
            print("[two_level_cache.py: save] Unknown savetype [{}]. quitting".format(saveType))
            exit()
        self.lookup_cache.save(self.payload_filename)

    def load_payload_filename(self,load_information):
        # 1. collect all the matches in a list
        ds_filenames,_ = self.lookup_cache.load_all_matches()
        if len(ds_filenames) is 0:
            return None
        if len(ds_filenames) == 1 and load_information is None:
            return ds_filenames[0]
        if len(ds_filenames) >= 2 and load_information is None:
            print("[TwoLevelCache] ERROR: two or more matching configs.")
            exit()
        # 2. use policy to determine which matches to use (we known "load_information" is not None)
        if load_information.mode is "regex":
            print(load_information.regex)
            match_info_list = []
            for filename in ds_filenames:
                print(filename)
                match_info = re.match(load_information.regex,filename)
                if match_info is not None:
                    return filename
        print("failed to match on load_information")
        return None

    def load(self,saveType="pickle",load_information=None):
        self.payload_filename = self.load_payload_filename(load_information)
        if self.payload_filename is None:
            return None
        if saveType == "pickle":
            payload = readPickle(self.payload_filename)
        elif saveType == "ndarray":
            payload = readNdarray(self.payload_filename)
        else:
            print("[two_level_cache.py: load] Unknown savetype [{}]. quitting".format(saveType))
            exit()
        return payload

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



