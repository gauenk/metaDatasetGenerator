import os,cPickle
import os.path as osp
from utils.cache import Cache

class RoidbCache():

    def __init__(self,root_dir,imdb_str,configToCompare):
        self.root_dir = root_dir
        self.cache_dir = osp.join(root,imdb_str)
        self.configToCompare = configToCompare
        lookup_filename = osp.join(cache_dir,'lookup.pkl')
        self.lookup_cache = Cache(lookup_filename,configToCompare)

    def construct_data_cache_config(self,cfg,ds_config):
        cacheDataCfg = edict()

        cacheDataCfg.task = cfg.TASK
        cacheDataCfg.subtask = cfg.SUBTASK

        cacheDataCfg.dataset_augmentation = edict() # primary config set 1
        cacheDataCfg.dataset_augmentation.bool_value = cfg.DATASET_AUGMENTATION.BOOL
        cacheDataCfg.dataset_augmentation.CONFIGS = cfg.DATASET_AUGMENTATION.CONFIGS # says: "what type of augmentations do we have?"
        cacheDataCfg.dataset_augmentation.percent_samples_augmented = cfg.DATASET_AUGMENTATION.N_SAMPLES # says: "how many samples are we augmenting?"
        cacheDataCfg.dataset_augmentation.bool_by_samples = cfg.dataset_augmentation.SAMPLE_BOOL_VECTOR # says: which samples are augmented?
        cacheDataCfg.dataset_augmentation.randomize = cfg.DATASET_AUGMENTATION.RANDOMIZE
        cacheDataCfg.dataset_augmentation.randomize_subset = cfg.DATASET_AUGMENTATION.RANDOMIZE_SUBSET

        cacheDataCfg.dataset = edict() # primary config set 2
        cacheDataCfg.dataset.subsample_bool = cfg.DATASETS.SUBSAMPLE_BOOL
        cacheDataCfg.dataset.annotation_class = cfg.DATASETS.ANNOTATION_CLASS
        cacheDataCfg.dataset.filter_empty_annotations = cfg.DATASETS.FILTER_EMPTY_ANNOTATIONS
        cacheDataCfg.dataset.size = cfg.DATASETS.SIZE #len(roidb)
        cacheDataCfg.dataset.classes = cfg.DATASETS.CLASSES #imdb.classes

        cacheDataCfg.misc = edict() # primary config set 3
        cacheDataCfg.misc.use_diff = ds_confi['use_diff']
        cacheDataCfg.misc.rpn_file = ds_confi['rpn_file']
        cacheDataCfg.misc.min_size = ds_confi['min_size']
        cacheDataCfg.misc.flatten_image_index = ds_confi['flatten_image_index']
        cacheDataCfg.misc.setID = ds_confi['setID']

        # TODO: put theses somewhere else
        # ? assert len(roidb) == cacheDataCfg.DATASET.SIZE
        # ? assert imdb.classes == cacheDataCfg.DATASET.CLASSES
    
    def save_cache(uuid_str,roidb):
        # cache the roidb
        self.roidb_filename = osp.join(cache_dir,'{}.pkl'.format(uuid_str))
        if os.exists(self.cache_dir) is False:
            os.makedirs(self.cache_dir)
        with open(self.roidb_filename, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        lookup.save(roidb_filename)
    

    
