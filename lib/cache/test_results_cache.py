import os,uuid,copy
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

class TestResultsCache(TwoLevelCache):

    def __init__(self,root_dir,cfg,imdb_config,roidb_settings,lookup_id):
        self.test_cache_config = self.construct_data_cache_config(cfg,imdb_config)
        super(TestResultsCache,self).__init__(root_dir,self.test_cache_config,roidb_settings,lookup_id,"test_results")

    def reset_dataset_augmentation(self,datasetAugmentationCfg):
        temporaryConfig = copy.deepcopy(self.testCacheConfig)
        self.set_dataset_augmentation(temporaryConfig,datasetAugmentationCfg)
        self.update_config(temporaryConfig)

    def set_dataset_augmentation(self,test_cache_config,datasetAugmentationCfg):
        test_cache_config.dataset_augmentation = edict() # primary config set 1
        test_cache_config.dataset_augmentation.bool_value = datasetAugmentationCfg.BOOL
        #test_cache_config.dataset_augmentation.configs = datasetAugmentationCfg.CONFIGS # says: "what type of augmentations do we have?"
        test_cache_config.dataset_augmentation.image_translate = datasetAugmentationCfg.IMAGE_TRANSLATE
        test_cache_config.dataset_augmentation.image_rotate = datasetAugmentationCfg.IMAGE_ROTATE
        test_cache_config.dataset_augmentation.image_crop = datasetAugmentationCfg.IMAGE_CROP
        test_cache_config.dataset_augmentation.image_flip = datasetAugmentationCfg.IMAGE_FLIP
        test_cache_config.dataset_augmentation.percent_augmentations_used = datasetAugmentationCfg.N_PERC # says: "how many of the possible augmentations should we use for each augmented sample?"
        test_cache_config.dataset_augmentation.percent_samples_augmented = datasetAugmentationCfg.N_SAMPLES # says: "how many samples are we augmenting?"
        # test_cache_config.dataset_augmentation.bool_by_samples = datasetAugmentationCfg.SAMPLE_BOOL_VECTOR # says: which samples are augmented?
        test_cache_config.dataset_augmentation.randomize = datasetAugmentationCfg.RANDOMIZE
        test_cache_config.dataset_augmentation.randomize_subset = datasetAugmentationCfg.RANDOMIZE_SUBSET
        

    def construct_data_cache_config(self,cfg,imdb_config):
        testCacheConfig = edict()

        testCacheConfig.id = "default_id"
        testCacheConfig.task = cfg.TASK
        testCacheConfig.subtask = cfg.SUBTASK
        testCacheConfig.modelInfo = cfg.modelInfo
        testCacheConfig.transform_each_sample = cfg.TRANSFORM_EACH_SAMPLE
        # we don't want to save the actual config within the modelInfo
        testCacheConfig.modelInfo.additional_input.info['activations'].cfg = None

        self.set_dataset_augmentation(testCacheConfig,cfg.DATASET_AUGMENTATION)
        testCacheConfig.dataset = edict() # primary config set 2
        testCacheConfig.dataset.name = cfg.DATASETS.CALLING_DATASET_NAME
        testCacheConfig.dataset.imageset = cfg.DATASETS.CALLING_IMAGESET_NAME
        testCacheConfig.dataset.config = cfg.DATASETS.CALLING_CONFIG
        testCacheConfig.dataset.subsample_bool = cfg.DATASETS.SUBSAMPLE_BOOL
        testCacheConfig.dataset.annotation_class = cfg.DATASETS.ANNOTATION_CLASS
        # testCacheConfig.dataset.size = cfg.DATASETS.SIZE #len(roidb) ## We can't use this because it is set by the unfiltered image_index variable
        # testCacheConfig.dataset.classes = cfg.DATASETS.CLASSES #imdb.classes; classes should already be filtered if they need to be ## can't use this it is unfiltered

        
        testCacheConfig.filters = edict() # primary config set 3
        testCacheConfig.filters.classes = cfg.DATASETS.FILTERS.CLASS
        testCacheConfig.filters.empty_annotations = cfg.DATASETS.FILTERS.EMPTY_ANNOTATIONS

        testCacheConfig.misc = edict() # primary config set 3
        testCacheConfig.misc.use_diff = imdb_config['use_diff']
        testCacheConfig.misc.rpn_file = imdb_config['rpn_file']
        testCacheConfig.misc.min_size = imdb_config['min_size']
        testCacheConfig.misc.flatten_image_index = imdb_config['flatten_image_index']
        testCacheConfig.misc.setID = imdb_config['setID']

        return testCacheConfig
        # TODO: put theses somewhere else
        # ? assert len(roidb) == testCacheConfig.DATASET.SIZE
        # ? assert imdb.classes == testCacheConfig.DATASET.CLASSES

    def print_dataset_summary_by_uuid(datasetConfigList,uuid_str):
        pass
