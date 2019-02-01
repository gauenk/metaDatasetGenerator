# --------------------------------------------------------
# CBAL
# Written by Kent Gauen
# --------------------------------------------------------

import os,glob,sys
import os.path as osp
import PIL
import pickle
import yaml,uuid
import numpy as np
from datasets.imdb import imdb
import scipy.sparse
from numpy import random as npr
from core.config import cfg,cfgData,cfgData_from_file,get_output_dir
from easydict import EasyDict as edict
from utils.base import *
from datasets.evaluators.regression import regressionEvaluator
from datasets.evaluators.classification import classificationEvaluator
from datasets.evaluators.bboxEvaluator import bboxEvaluator
from datasets.imageReader.pathReader import pathReader
from datasets.annoReader.xmlReader import xmlReader
from datasets.annoReader.txtReader import txtReader
from datasets.annoReader.jsonReader import jsonReader
from datasets.annoReader.pathDerivedReader import pdReader
from datasets.data_utils.roidb_utils import *
from datasets.data_loader import DataLoader
from cache.roidb_cache import RoidbCache


class DatasetObject(imdb):
    """Infrastructure of pipes to the dataset locations."""

    def __init__(self, datasetName, imageSet, configName, path_to_imageSets=None,cacheStrModifier=None,load_roidb=True):
        imdb.__init__(self, datasetName)
        cfg.DATASETS.CALLING_DATASET_NAME = datasetName
        cfg.DATASETS.CALLING_IMAGESET_NAME = imageSet
        cfg.DATASETS.CALLING_CONFIG = configName
        cfg.DATASETS.CALLING_IMDB_STR = '{}_{}_{}'.format(datasetName,imageSet,configName)
        self.imdb_str = '{}_{}_{}'.format(datasetName,imageSet,configName)
        self._imdb_str = '{}-{}-{}'.format(datasetName,imageSet,configName)
        self.roidb_loaded = False
        self.data_loader = False
        self._cacheStrModifier = cacheStrModifier
        self._path_to_imageSets = path_to_imageSets
        self._local_path = os.path.dirname(__file__)
        self._datasetName = datasetName
        self._configName = configName
        self._image_set = imageSet
        self._image_index = [] 
        self._convertIdToCls = None
        self._obj_proposer = 'gt'
        self._roidb = None
        self._roidb_handler = self.gt_roidb
        self._roidbSize = []
        self.is_image_index_flattened = False
        cfg.DATASETS.IS_IMAGE_INDEX_FLATTENED = False
        self._parseDatasetFile()
        self.roidb_cache_fieldname = 'object'
        self.roidb_cache = RoidbCache(self.cache_path,self._imdb_str,cfg,self.config,None,'dataset_object',self.roidb_cache_fieldname)
        if load_roidb:
            self.gt_roidb()

    def _parseDatasetFile(self):
        self._resetDataConfig()
        self._setupConfig()
        fn = osp.join(self._local_path,"ymlDatasets", cfg.PATH_YMLDATASETS,self._datasetName + ".yml")
        cfgData_from_file(fn)
        self._pathResults = cfgData['PATH_TO_RESULTS']
        assert self._datasetName == cfgData['EXP_DATASET'], "dataset name is not correct."
        self._set_classes(cfgData['CLASSES'],cfgData['CONVERT_TO_PERSON'],cfgData['ONLY_PERSON'])
        self._num_classes = len(self._classes)
        self._path_root = cfgData['PATH_ROOT']
        self._compID = cfgData['COMPID']
        if self._cacheStrModifier:
            self._cachedir = os.path.join(self._path_root,\
                                      'annotations_cache',\
                                      self._image_set+"_"+self._cacheStrModifier)
        else:
            self._cachedir = os.path.join(self._path_root,\
                                          'annotations_cache',
                                          self._image_set)
        if self._path_to_imageSets is None:
            self._path_to_imageSets = cfgData['PATH_TO_IMAGESETS']
        if not self._checkImageSet():
            raise ValueError("imageSet path {} doesn't exist".format(self._imageSetPath))

        self._image_index = None#self._load_image_index()

        self._set_id_to_cls()

        self.annoReader = self._createAnnoReader(cfgData['PATH_TO_ANNOTATIONS'],
                                                 cfgData['ANNOTATION_TYPE'],
                                                 cfgData['PARSE_ANNOTATION_REGEX'],
                                                 cfgData['CONVERT_TO_PERSON'],
                                                 cfgData['USE_IMAGE_SET'],
        )
        self.evaluator = self._createEvaluator(cfgData['PATH_TO_ANNOTATIONS'])
        self.imgReader = self._createImgReader(cfgData['PATH_TO_IMAGES'],
                                               cfgData['IMAGE_TYPE'],
                                               cfgData['USE_IMAGE_SET'],
        )
        self._image_index = self._load_image_index()
        cfg.DATASETS.SIZE = len(self._image_index)

    def _resetDataConfig(self):
        cfgData.CONVERT_ID_TO_CLS_FILE = None
        cfgData.USE_IMAGE_SET = None
        
    def _set_id_to_cls(self):
        convertIdtoCls_filename = cfgData['CONVERT_ID_TO_CLS_FILE']
        if convertIdtoCls_filename is not None:
            path = osp.join(self._local_path,
                            "ymlDatasets",
                            convertIdtoCls_filename)
            with open(path,"r") as f:
                self._convertIdToCls = edict(yaml.load(f))

    def _setupConfig(self):
        fn = osp.join(self._local_path,"ymlConfigs" ,self._configName + ".yml")
        with open(fn, 'r') as f:
            yaml_cfg = edict(yaml.load(f))
        # fn = osp.join(self._local_path,"ymlConfigs" ,
        #               yaml_cfg['CONFIG_DATASET_INDEX_DICTIONARY_PATH'])
        # with open(fn, 'r') as f:
        #     setID = edict(yaml.load(f))[self._datasetName]
        try:
            setID = cfg.DATASET_NAMES_ORDERED.index(self._datasetName)
        except:
            setID = cfg.DATASET_NAMES_ORDERED.index(self._datasetName.split("_")[0])
        cfg.DATASETS.HAS_BBOXES = yaml_cfg['CONFIG_HAS_BBOXES']
        if 'ANNOTATION_CLASS' in yaml_cfg.keys():
            cfg.DATASETS.ANNOTATION_CLASS = yaml_cfg['ANNOTATION_CLASS']
        self.config = {'cleanup'     : yaml_cfg['CONFIG_CLEANUP'],
                       'use_salt'    : yaml_cfg['CONFIG_USE_SALT'],
                       'use_diff'    : yaml_cfg['CONFIG_USE_DIFFICULT'],
                       'rpn_file'    : yaml_cfg['CONFIG_RPN_FILE'],
                       'min_size'    : yaml_cfg['CONFIG_MIN_SIZE'],
                       'flatten_image_index' : yaml_cfg['CONFIG_FLATTEN_IMAGE_INDEX'],
                       'setID'       : setID}
        self.config = edict(self.config)
        
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        if type(i) is int:
            index = self._image_index[i]
        elif type(i) is str:
            index = i
        return self.imgReader.image_path_from_index(index)

    def image_index_at(self,i):
        return self._image_index[i]
        
    def _set_classes(self,classFilename,convertToPersonList,onlyPerson):
        _classes = self._load_classes(classFilename)
        if cfg.DATASETS.ANNOTATION_CLASS == 'object_detection':
            assert _classes[0] == "__background__","Background class must be first index"
        elif _classes[0] == "__background__":
            del _classes[0]
        self._classes = _classes
        self._original_classes = self._classes        

    def create_data_loader(self,cfg,correctness_records,al_net):
        loadConfig = edict()
        loadConfig.activation_sample = edict()
        loadConfig.activation_sample.bool_value = cfg.LOAD_METHOD == 'aim_data_layer' #cfg.TEST.INPUTS.AV_IMAGE
        loadConfig.activation_sample.net = al_net
        loadConfig.activation_sample.field = 'image' # 'image' or 'avImage'
        load_rois_bool = (cfg.TRAIN.OBJ_DET.HAS_RPN is False) and (cfg.TASK == 'object_detection')
        loadConfig.load_rois_bool = load_rois_bool # cfg.TEST.INPUTS.ROIS
        loadConfig.target_size = cfg.TRAIN.SCALES[0]
        loadConfig.target_size_siamese = cfg.TRAIN.SCALES
        loadConfig.cropped_to_box_bool = False
        loadConfig.cropped_to_box_index = 0
        loadConfig.dataset_means = cfg.PIXEL_MEANS
        loadConfig.max_sample_single_dimension_size = cfg.TRAIN.MAX_SIZE
        loadConfig.load_fields = ['data']
        loadConfig.preprocess_image = True
        # transform_sample_label_by_augmentation = (cfg.SUBTASK == 'angles')
        #loadConfig.replace_labels_after_augmentation_file = cfg.REPLACE_LABELS_AFTER_AUGMENTATION_FILE
        loadConfig.additional_input = cfg.modelInfo.additional_input
        loadConfig.sample_type = "regularOldImage"
        loadConfig.siamese = False
        self.data_loader_config = loadConfig
        ds_loader = DataLoader(self,correctness_records,cfg.DATASET_AUGMENTATION,cfg.TRANSFORM_EACH_SAMPLE,self.roidb_cache)
        ds_loader.set_dataset_loader_config(loadConfig)
        self.data_loader = ds_loader
        return ds_loader

    def convertClassToPerson(_classes,convertToPersonList):
        # handle class names that are not quite person, but should be
        if convertToPersonList is None:
            return _classes

        anyCommonClasses = check_list_equal_any(convertToPersonList,_classes)
        if anyCommonClasses is False:
            return _classes
        else:
            # ensure person is in classes
            classes = _classes[:]
            if "person" not in classes:
                classes.append("person")
            # remove all classes in "converToPerson" list
            for _class in _classes:
                if _class in convertToPersonList:
                    classes.remove(_class)
            _classes = classes
        return classes


    def _createAnnoReader(self,annoPath,annoType,cleanRegex,convertToPersonList,useImageSet):
        path = annoPath
        if useImageSet:
            path = osp.join(path,self._image_set)
        if annoType == "xml": return xmlReader(path,self.classes,self._datasetName,
                                               self.config['setID'],
                                               convertToPerson=convertToPersonList,
                                               convertIdToCls = self._convertIdToCls,
                                               is_image_index_flattened = self.is_image_index_flattened)
        elif annoType == "txt": return txtReader(path,self.classes,self._datasetName,
                                                 self.config['setID'],cleanRegex=cleanRegex,
                                                 convertToPerson=convertToPersonList,
                                                 convertIdToCls = self._convertIdToCls,
                                                 is_image_index_flattened =  self.is_image_index_flattened)
        elif annoType == "cls_txt": return txtReader(path,self.classes,self._datasetName,
                                                     self.config['setID'],anno_type='cls',
                                                     is_image_index_flattened =  self.is_image_index_flattened)
        elif annoType == "cls_derived": return pdReader(path,self.classes,self._datasetName,
                                                        self.config['setID'],
                                                        convertIdToCls=self._convertIdToCls,
                                                        is_image_index_flattened =  self.is_image_index_flattened)
        elif annoType == "json": return jsonReader(path,self.classes,self._datasetName,
                                                   self.config['setID'],self._image_set,
                                                   convertToPerson=convertToPersonList,
                                                   convertIdToCls = self._convertIdToCls,
                                                   is_image_index_flattened =  self.is_image_index_flattened)

    def _createImgReader(self,imgPath,imgType,useImageSet):
        path = imgPath
        if useImageSet:path = osp.join(path,self._image_set)
        return pathReader(path,imgType,is_image_index_flattened = self.is_image_index_flattened)

    def _createEvaluator(self,annoPath):
        if self.config['use_salt']: self._salt = str(uuid.uuid4())
        else: self._salt = None
        cachedir = self._cachedir
        if not osp.isdir(cachedir):
            os.makedirs(cachedir)
        if cfg.DATASETS.ANNOTATION_CLASS == 'object_detection':
            return bboxEvaluator(self._datasetName,self.classes,
                             self._compID, self._salt,
                             cachedir, self._imageSetPath,
                             self._image_index,annoPath,
                             self.load_annotation)
        elif cfg.TASK == 'classification':
            return classificationEvaluator()
        elif cfg.TASK == 'regression':
            return regressionEvaluator()
        else:
            print("\n\n\nNo Evaluator Included\n\n\n")
            return None

    def _update_classes(self,new_class_list):
        new_class_list = [str(cls) for cls in new_class_list]
        self._original_classes = self._classes
        if self.evaluator:
            self.evaluator._class_convert = [self._classes.index(str(cls)) for cls in new_class_list]
            self.evaluator._classes = new_class_list
        if self.annoReader:
            self.annoReader.num_classes = len(new_class_list)
            self.annoReader._classToIndex = self.annoReader._create_classToIndex(new_class_list)
            self.annoReader.class_filter = self.class_filter
        self._classes = new_class_list
        self._num_classes = len(self._classes)

    def _update_image_index(self,newImageIndex):
        self._image_index = newImageIndex
        if self.imgReader:
            self.imgReader._image_index = newImageIndex
        if self.data_loader:
            self.image_index = newImageIndex
        
    def _update_is_image_index_flattened(self,state):
        self.is_image_index_flattened = state
        cfg.DATASETS.IS_IMAGE_INDEX_FLATTENED = state
        if self.annoReader: self.annoReader._is_image_index_flattened = state
        if self.imgReader: self.imgReader._is_image_index_flattened = state
        
    def _update_path_to_imagesets(self,new_path_to_imagesets):
        self._path_to_imagesets = new_path_to_imagesets
        
    def _load_image_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        uses: _path_root, _image_set
        """
        image_set_file = osp.join(self._path_to_imageSets,self._image_set + '.txt')
        assert os.path.exists(image_set_file),'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip().split()[0] for x in f.readlines()]
        self.checkImageIndex(image_set_file,image_index)
        image_index = sorted(image_index) # this line maintains order forever
        if self.config['flatten_image_index']:
            image_index = self.apply_flatten_image_index(image_index)
        self._update_image_index(image_index)
        return image_index
        
    def checkImageIndex(self,image_set_file,image_index):
        if len(image_index) == 0:
            print("Likely Error with ImageIndex. len(image_index) = 0")
            print("image_set_file: {}".format(image_set_file))
            sys.exit()
        # if cfg.DATASETS.ANNOTATION_CLASS == "classification" and self.config['flatten_image_index']:
        #     print("\n\n\n\nPossible Error: ANNOTATION_CLASS is 'classification' while 'flatten_image_index' is True\n\n\n\n\n")

    
    def apply_flatten_image_index(self,image_index):
        newImageIndex = []
        for idx,image_id in enumerate(image_index):
            image_count = self.count_bboxes_at(idx,image_id)
            id_list = [ image_id + "_{}".format(jdx) for jdx in range(image_count) ]
            newImageIndex.extend(id_list)
        image_index = newImageIndex
        self._update_is_image_index_flattened(True)
        return image_index

    def count_bboxes_at(self,i,index=None):
        """
        Return the number of bounding boxes for image i in the image sequence.
        """
        if index is None:
            index = self._image_index[i]
        return len(self.annoReader.load_annotation(index)['boxes'])

    def _load_classes(self, classFilename):
        with open(classFilename,"r") as f:
            classes = [line.rstrip() for line in f]
        return classes

    def update_dataset_augmentation(self,dataset_augmentation):
        if self.data_loader:
            self.data_loader.update_dataset_augmentation(dataset_augmentation)

    def evaluation_results(self, ds_loader, agg_model_output, output_dir):
        self.evaluator.set_evaluation_parameters(self, ds_loader, agg_model_output, output_dir)
        return self.evaluator.evaluation_results
        
    def evaluate_model_inference(self, ds_loader, agg_model_output, output_dir,**kwargs):
        """
        detection_object is a diction with two keys:
        "all_boxes" and "im_rotates_all"

        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5

        im_rotates_all is a list of the rotation matricies for transforming
        the groundtruth polygon into a rotated version of the polygon to compare 
        with the predicted polygon.
        """
        if ds_loader is None:
            print("[ds_obj]: ds_loader was None. we are creating one for evaluation.")
            ds_loader = self.create_data_loader(cfg,None,None)
        self.evaluator.set_evaluation_parameters(self, ds_loader, agg_model_output, output_dir, **kwargs)
        self.evaluator.evaluate_detections()
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_results_file_template().format(cls)
                #os.remove(filename)

    def _get_results_file_template(self):
        # example: VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._compID + self._salt + '_det_' + self._image_set + '_{:s}.txt'
        path = osp.join(self._pathResults,filename)
        return path

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cfg.DATASETS.FILTERS.CLASS_INCLUSION_LIST = []   # NOTE! THIS FEATURE IS INCOMPLETE!!
        
        
        self.filters = edict()
        self.filters.class_bool = cfg.DATASETS.FILTERS.CLASS
        self.class_filter = None
        self.class_filter = edict()
        self.class_filter.check = cfg.DATASETS.FILTERS.CLASS
        self.class_filter.original_names = self._classes
        self.annoReader.class_filter = self.class_filter
        set_name = self._imdb_str.split('-')[0]
        if cfg.DATASETS.FILTERS.CLASS:
            self.class_filter.new_names = cfg.DATASETS.FILTERS.CLASS_INCLUSION_LIST_BY_DATASET[set_name]
        else:
            self.class_filter.new_names = False
    
        
        if self.filters.class_bool:
            self.class_filter = edict()
            self.class_filter.check = True
            self.class_filter.original_names = self._classes
            set_name = self._imdb_str.split('-')[0]
            self.class_filter.new_names = cfg.DATASETS.FILTERS.CLASS_INCLUSION_LIST_BY_DATASET[set_name]

        self.filters.empty_annotations = cfg.DATASETS.FILTERS.EMPTY_ANNOTATIONS
        self.config.subsample_bool = cfg.DATASETS.SUBSAMPLE_BOOL
        self.config.subsample_size = cfg.DATASETS.SUBSAMPLE_SIZE

        loaded_payload = self.roidb_cache.load(self.roidb_cache_fieldname)
        self.roidb_loaded = False
        if loaded_payload is None:
            gt_roidb = [self.load_annotation(index) for index in self._image_index]
            gt_roidb, image_index = self.apply_roidb_filters(gt_roidb,self._image_index)
            payload = [gt_roidb, image_index]
            self.roidb_loaded = True
            self.roidb_cache.save(self.roidb_cache_fieldname,payload)
        else:
            self.roidb_loaded = True
            gt_roidb, image_index = loaded_payload
            gt_roidb, image_index = self.apply_roidb_filters(gt_roidb,image_index)
        self._update_image_index(image_index)
        self._update_roidb(gt_roidb)
        return gt_roidb

    def apply_roidb_filters(self,gt_roidb,image_index):

        class_inclusion_list = self.class_filter.new_names

        if self.roidb_loaded is False:
            if self.filters.empty_annotations is True:
                gt_roidb, image_index = filterSampleWithEmptyAnnotations(gt_roidb,image_index)
            if self.filters.class_bool is True:
                gt_roidb, image_index = filterImagesByClass(gt_roidb,image_index,self.class_filter)
            if self.config.subsample_bool is True:
                gt_roidb, image_index = subsample_roidb(gt_roidb,image_index,self.config.subsample_size)

        if self.filters.class_bool is True:
            self.class_conversion_dict = createClassConversionDict(class_inclusion_list,self._classes)
            self._update_classes(class_inclusion_list)

        return gt_roidb, image_index
                    
    def _update_roidb(self,roidb):
        self._roidb = roidb
        self.roidb_size = len(roidb)
        if self.data_loader:
            self.data_loader.roidb = roidb

    def convert_class_name_to_class_index(self,class_name):
        if self.filters.class_bool:
            return self.class_conversion_dict[class_name]
        else:
            return self._classes.index(class_name)
            
    def convert_class_index_to_class_name(self,class_index,is_class_index_converted=False):
        if self.filters.class_bool and is_class_index_converted is False:
            return self._original_classes[class_index]
        else:
            return self._classes[class_name]
        
    def load_annotation(self,index):
        sample = self.annoReader.load_annotation(index)
        cfg.DATASET_AUGMENTATION.DEFAULT_BOOL = False
        cfg.DATASET_AUGMENTATION.DEFAULT_INDEX = 0
        sample['aug_bool'] = cfg.DATASET_AUGMENTATION.DEFAULT_BOOL
        sample['aug_index'] = cfg.DATASET_AUGMENTATION.DEFAULT_INDEX
        return sample

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

    def compute_size_along_roidb(self):
        if self.roidb_loaded is False:
            raise ValueError("roidb must be loaded before 'compute_size_along_roidb' can be run")
        self._roidbSize = []

        if cfg.DATASETS.ANNOTATION_CLASS == "object_detection":
           self._roidbSize.append(len(self.roidb[0]['gt_classes']))
           for image in self.roidb[1:]:
               newSize = self._roidbSize[-1] + len(image['gt_classes'])
               self._roidbSize.append(newSize)
        elif cfg.DATASETS.ANNOTATION_CLASS == "classification" or cfg.DATASETS.ANNOTATION_CLASS == "regression":
            self._roidbSize = np.arange(len(self._image_index)) + 1
        else:
            print("ERROR: the cfg.DATASETS.ANNOTATION_CLASS is not recognized.")
            sys.exit()
    
    def __str__(self):
        return self.imdb_str

    def get_roidb_labels(self):
        gt_labels = np.zeros((self.roidb_size,self.num_classes),dtype=np.uint8)
        for index,sample in enumerate(self.roidb):
            gt_labels[index] = sample['gt_classes']
        return gt_labels
