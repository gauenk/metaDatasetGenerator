# --------------------------------------------------------
# CBAL
# Written by Kent Gauen
# --------------------------------------------------------

import os,glob,sys
import os.path as osp
import PIL
import cPickle
import yaml,uuid
import numpy as np
from datasets.imdb import imdb
import scipy.sparse
from numpy import random as npr
from core.config import cfg,cfgData,cfgData_from_file
from easydict import EasyDict as edict
from datasets.evaluators.classification import classificationEvaluator
from datasets.evaluators.bboxEvaluator import bboxEvaluator
from datasets.imageReader.rawReader import rawReader
from datasets.annoReader.xmlReader import xmlReader
from datasets.annoReader.txtReader import txtReader
from datasets.annoReader.jsonReader import jsonReader
from datasets.annoReader.pathDerivedReader import pdReader

class RepoImdb(imdb):
    """Image database."""

    def __init__(self, datasetName, imageSet, configName, path_to_imageSets=None,cacheStrModifier=None):
        imdb.__init__(self, datasetName)
        cfg.DATASETS.CALLING_DATASET_NAME = datasetName
        cfg.DATASETS.CALLING_IMAGESET_NAME = imageSet
        cfg.DATASETS.CALLING_CONFIG = configName
        self._cacheStrModifier = cacheStrModifier
        self._path_to_imageSets = path_to_imageSets
        self._local_path = os.path.dirname(__file__)
        self._datasetName = datasetName
        self._configName = configName
        self._image_set = imageSet
        print(self._image_set)
        self._image_index = [] 
        self._convertIdToCls = None
        self._obj_proposer = 'gt'
        self._roidb = None
        self._roidb_handler = self.gt_roidb
        self._roidbSize = []
        self.is_image_index_flattened = False
        cfg.DATASETS.IS_IMAGE_INDEX_FLATTENED = False
        self._parseDatasetFile()

    def _parseDatasetFile(self):
        self._resetDataConfig()
        self._setupConfig()
        print(self.config)
        fn = osp.join(self._local_path,
                      "ymlDatasets", cfg.PATH_YMLDATASETS,
                      self._datasetName + ".yml")
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
        
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        # if self._roidb is None:
        #     self._roidb = self.roidb_handler()
        if type(i) is int:
            index = self._image_index[i]
        elif type(i) is str:
            index = i
        return self.imgReader.image_path_from_index(index)

    def image_index_at(self,i):
        return self._image_index[i]
        
    def _set_classes(self,classFilename,convertToPerson,onlyPerson):
        _classes = self._load_classes(classFilename)
        print(_classes)
        if cfg.DATASETS.ANNOTATION_CLASS == 'object_detection':
            assert _classes[0] == "__background__","Background class must be first index"
        
        if onlyPerson and cfg.DATASETS.ANNOTATION_CLASS == 'object_detection':
            _classes = ["__background__","person"]
        elif onlyPerson:
            _classes = ["person"]

        if convertToPerson is not None and not onlyPerson:
            # ensure person is in classes
            classes = _classes[:]
            if "person" not in classes:
                classes.append("person")
            # remove all classes in "converToPerson" list
            for _class in _classes:
                if _class in convertToPerson:
                    classes.remove(_class)
            _classes = classes
        self._classes = _classes        

    def _createAnnoReader(self,annoPath,annoType,cleanRegex,convertToPerson,useImageSet):
        path = annoPath
        if useImageSet:
            path = osp.join(path,self._image_set)
        if annoType == "xml": return xmlReader(path,self.classes,self._datasetName,
                                               self.config['setID'],
                                               convertToPerson=convertToPerson,
                                               convertIdToCls = self._convertIdToCls,
                                               is_image_index_flattened = self.is_image_index_flattened)
        elif annoType == "txt": return txtReader(path,self.classes,self._datasetName,
                                                 self.config['setID'],cleanRegex=cleanRegex,
                                                 convertToPerson=convertToPerson,
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
                                                   convertToPerson=convertToPerson,
                                                   convertIdToCls = self._convertIdToCls)
                                                   # is_image_index_flattened =  self.is_image_index_flattened)

    def _createImgReader(self,imgPath,imgType,useImageSet):
        path = imgPath
        if useImageSet:path = osp.join(path,self._image_set)
        return rawReader(path,imgType,is_image_index_flattened = self.is_image_index_flattened)

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
        elif cfg.DATASETS.ANNOTATION_CLASS == 'classification':
            return classificationEvaluator(self._datasetName,self.classes,
                             self._compID, self._salt,
                             cachedir, self._imageSetPath,
                             self._image_index,annoPath,
                             self.load_annotation)
        else:
            print("\n\n\nNo Evaluator Included\n\n\n")
            return None

    def _update_classes(self,new_class_list):
        if self.evaluator:
            self.evaluator._class_convert = [self._classes.index(str(cls)) for cls in new_class_list]
            self.evaluator._classes = new_class_list
        if self.annoReader:
            self.annoReader.num_classes = len(new_class_list)
            self.annoReader._classToIndex = self.annoReader._create_classToIndex(new_class_list)
        self._classes = new_class_list
        self._num_classes = len(self._classes)

    def _update_image_index(self,newImageIndex):
        self._image_index = newImageIndex
        if self.imgReader: self.imgReader._image_index = newImageIndex
        if self.evaluator: self.evaluator.image_index = newImageIndex
        
    def _update_is_image_index_flattened(self,state):
        self.is_image_index_flattened = state
        cfg.DATASETS.IS_IMAGE_INDEX_FLATTENED = state
        if self.annoReader: self.annoReader._is_image_index_flattened = state
        if self.evaluator: self.evaluator._is_image_index_flattened = state
        if self.imgReader: self.imgReader._is_image_index_flattened = state
        
    def _update_path_to_imagesets(self,new_path_to_imagesets):
        self._path_to_imagesets = new_path_to_imagesets
        
    def _load_image_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        uses: _path_root, _image_set
        """
        image_set_file = osp.join(self._path_to_imageSets,
                                  self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip().split()[0] for x in f.readlines()]
        self._image_index = image_index
        self.checkImageIndex(image_set_file)
        if cfg.DATASETS.ANNOTATION_CLASS == "classification" and self.config['flatten_image_index']:
            print("\n\n\n Possible Error: ANNOTATION_CLASS is 'classification'\
            while 'flatten_image_index' is True\n\n\n\n")
        if self.config['flatten_image_index']:
            newImageIndex = []
            for idx,image_id in enumerate(image_index):
                image_count = self.count_bboxes_at(idx)
                id_list = [ image_id + "_{}".format(jdx) for jdx in range(image_count) ]
                newImageIndex.extend(id_list)
            image_index = newImageIndex
            self._update_is_image_index_flattened(True)
        self._update_image_index(image_index)
        return image_index
    
    def checkImageIndex(self,image_set_file):
        image_index = self._image_index
        if len(image_index) == 0:
            print("Likely Error with ImageIndex. len(image_index) = 0")
            print("image_set_file: {}".format(image_set_file))
            sys.exit()

    def count_bboxes_at(self,i):
        """
        Return the number of bounding boxes for image i in the image sequence.
        """
        index = self._image_index[i]
        return len(self.annoReader.load_annotation(index)['boxes'])

    def _load_classes(self, classFilename):
        with open(classFilename,"r") as f:
            classes = [line.rstrip() for line in f]
        return classes

    def evaluate_detections(self, detection_object, output_dir=None):
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
        self.evaluator.evaluate_detections(detection_object,output_dir)
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

    def _get_cache_filename(self):
        cache_file = osp.join(self.cache_path,'{}_{}_{}'.format(self.name,self._image_set,self._configName))
        if cfg.DATASET_AUGMENTATION.BOOL: cache_file += '_aug{}Perc'.format(int(100*cfg.DATASET_AUGMENTATION.N_SAMPLES))
        classFilterBool = (cfg.DATASETS.ANNOTATION_CLASS == "classification" and cfg.DATASETS.FILER.CLASS)
        if classFilterBool: cache_file += '_nClasses{}'.format(len(cfg.DATASETS.FILTERS.CLASS_INCLUSION_LIST))
        if self._cacheStrModifier: cache_file += '_{}'.format(self._cacheStrModifier)
        cache_file += '_gt_roidb.pkl'
        return cache_file
            
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        if self.is_image_index_flattened is False and self.config['flatten_image_index'] is True:
            print("Image Index has not yet been flattened")
            sys.exit()

        cache_file = self._get_cache_filename()
        print(cache_file)
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb_info = cPickle.load(fid)
                roidb = roidb_info["gt_roidb"]
                filtered_image_index = roidb_info["fii"]
                new_class_list = roidb_info['fcls']
                self._update_classes(new_class_list)
                self.is_image_index_flattened = roidb_info['is_image_index_flattened']
                if self.config['flatten_image_index'] != self.is_image_index_flattened:
                    print("\n\n\nERROR: Flattened image index loaded but not asked for. Exiting.")
                    sys.exit()
                self._update_is_image_index_flattened(self.is_image_index_flattened)
                print(len(roidb),len(filtered_image_index),len(self.image_index))
                if self.config['flatten_image_index']:
                    print("loading: imdb image index flattened")
                else:
                    print("loading: imdb image index not flattened")
                if filtered_image_index:
                    print("loading a filtered imdb")
                    self._update_image_index(list(filtered_image_index))
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self.load_annotation(index)
                    for index in self._image_index]

        # filter samples with no bounding box annotations:
        print("filtering empty samples from roidb")
        class_inclusion_list = self._classes
        if cfg.DATASETS.ANNOTATION_CLASS == "object_detection":
            gt_roidb,filtered_image_index = self.filterEmptyBoundingBoxSamples(gt_roidb)
        elif cfg.DATASETS.ANNOTATION_CLASS == "classification" and cfg.DATASETS.FILTERS.CLASS:
            gt_roidb,filtered_image_index = self.filterImagesByClass(cfg.DATASETS.FILTERS.CLASS_INCLUSION_LIST,gt_roidb)
            class_inclusion_list = cfg.DATASETS.FILTERS.CLASS_INCLUSION_LIST
        else:
            gt_roidb,filtered_image_index = gt_roidb,self._image_index

        # --> SUBSAMPLE THE DATA <--
        if cfg.DATASETS.SUBSAMPLE_BOOL:
            original_size = npr.permutation(len(gt_roidb))
            shuffled_indices = npr.permutation(original_size)[:cfg.DATASETS.SUBSAMPLE_SIZE]
            gt_roidb = [gt_roidb[index] for index in shuffled_indices]
            filtered_image_index = [filtered_image_index[index] for index in shuffled_indices]
            self._update_image_index(list(filtered_image_index))

        with open(cache_file, 'wb') as fid:
            cPickle.dump({"gt_roidb":gt_roidb,"fii":filtered_image_index,'is_image_index_flattened':self.is_image_index_flattened,'fcls':class_inclusion_list}, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb
        

    def removeListFromGtRoidb(self,gt_roidb,toRemove):
        numFiltered = len(toRemove)
        filtered_image_index = list(self._image_index)
        for idx in sorted(toRemove,reverse=True):
            del gt_roidb[idx]
            del filtered_image_index[idx]
        print("filtered {} samples".format(numFiltered))
        self._update_image_index(list(filtered_image_index))
        return gt_roidb,filtered_image_index
        
    def filterEmptyBoundingBoxSamples(self,gt_roidb):
        toRemove = []
        print("[repo_imdb.py: filterEmptyBoundingBoxSamples]")
        if cfg.DATASETS.ANNOTATION_CLASS == "object_detection":
            for idx,sample in enumerate(gt_roidb):
                if len(sample['gt_classes']) == 0: toRemove.append(idx)
            return self.removeListFromGtRoidb(gt_roidb,toRemove)
        return gt_roidb,self._image_index

    def filterImagesByClass(self,class_inclusion_list,gt_roidb):
        print("[repo_imdb.py: filterImagesByClass]")
        if len(class_inclusion_list) == 0: return gt_roidb,self._image_index
        toRemove = []
        if cfg.DATASETS.ANNOTATION_CLASS == "classification":
            for idx,sample in enumerate(gt_roidb):
                if sample['gt_classes'][0] not in class_inclusion_list: toRemove.append(idx)
            self._update_classes(class_inclusion_list)
            return self.removeListFromGtRoidb(gt_roidb,toRemove)
        return gt_roidb,self._image_index
                    
                    
    def load_annotation(self,index):
        return self.annoReader.load_annotation(index)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

    def compute_size_along_roidb(self):
        if self.roidb is None:
            raise ValueError("roidb must be loaded before 'compute_size_along_roidb' can be run")
        self._roidbSize = []

        if cfg.DATASETS.ANNOTATION_CLASS == "object_detection":
           self._roidbSize.append(len(self.roidb[0]['gt_classes']))
           for image in self.roidb[1:]:
               newSize = self._roidbSize[-1] + len(image['gt_classes'])
               self._roidbSize.append(newSize)
        elif cfg.DATASETS.ANNOTATION_CLASS == "classification":
            self._roidbSize = np.arange(len(self._image_index)) + 1
        else:
            print("ERROR: the cfg.DATASETS.ANNOTATION_CLASS is not recognized.")
            sys.exit()
