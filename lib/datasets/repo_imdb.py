# --------------------------------------------------------
# CBAL
# Written by Kent Gauen
# --------------------------------------------------------

import os
import os.path as osp
import PIL
import pickle
import yaml,uuid
import numpy as np
from datasets.imdb import imdb
import scipy.sparse
from core.config import cfg
from easydict import EasyDict as edict
from datasets.evaluators.bboxEvaluator import bboxEvaluator
from datasets.imageReader.rawReader import rawReader
from datasets.annoReader.xmlReader import xmlReader

class RepoImdb(imdb):
    """Image database."""

    def __init__(self, datasetName, imageSet, configName):
        imdb.__init__(self, datasetName)
        self._local_path = os.path.dirname(__file__)
        self._datasetName = datasetName
        self._configName = configName
        self._image_set = imageSet
        self._image_index = [] 
        self._parseDatasetFile()
        self._obj_proposer = 'gt'
        self._roidb = None
        self._roidb_handler = self.gt_roidb

    def _parseDatasetFile(self):
        self._setupConfig()
        fn = osp.join(self._local_path,"ymlDatasets" ,self._datasetName + ".yml")
        with open(fn, 'r') as f:
            yaml_cfg = edict(yaml.load(f))
        assert self._datasetName == yaml_cfg['EXP_DATASET'], "dataset name is not correct."
            
        self._classes = self._load_classes(yaml_cfg['CLASSES'])
        self._num_classes = len(self._classes)
        self._path_root = yaml_cfg['PATH_ROOT']

        self._path_to_imageSets = yaml_cfg['PATH_TO_IMAGESETS']
        if not self._checkImageSet():
            raise ValueError("imageSet path {} doesn't exist")

        self._image_index = self._load_image_index()

        self.evaluator = self._createEvaluator(yaml_cfg['COMPID'])

        self.imgReader = self._createImgReader(yaml_cfg['PATH_TO_IMAGES'],
                                                 yaml_cfg['IMAGE_TYPE'])

        self.annoReader = self._createAnnoReader(yaml_cfg['PATH_TO_ANNOTATIONS'],
                                                 yaml_cfg['ANNOTATION_TYPE'],
                                                 yml_cfg['PARSE_ANNOTATION_REGEX'])

    def _setupConfig(self):
        fn = osp.join(self._local_path,"ymlConfigs" ,self._configName + ".yml")
        with open(fn, 'r') as f:
            yaml_cfg = edict(yaml.load(f))
        self.config = {'cleanup'     : yaml_cfg['CONFIG_CLEANUP'],
                       'use_salt'    : yaml_cfg['CONFIG_USE_SALT'],
                       'use_diff'    : yaml_cfg['CONFIG_USE_DIFFICULT'],
                       'rpn_file'    : yaml_cfg['CONFIG_RPN_FILE'],
                       'min_size'    : yaml_cfg['CONFIG_MIN_SIZE']}
        
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.imgReader.image_path_at(i)

    def _checkImageSet(self):
        self._imageSetPath = osp.join(self._path_to_imageSets, self._image_set + ".txt")
        if osp.exists(self._imageSetPath) == True:
            return 1
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        print("imageSet error:")
        for imageset in glob.glob(self._path_to_imageSets):
            print(imageset)
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        return 0

    def _createAnnoReader(self,annoPath,annoType,cleanRegex):
        if annoType == "xml": return xmlReader(annoPath,self.classes,self._datasetName)
        elif annoType == "txt": return xmlReader(annoPath,self.classes,\
                                                 self._datasetName,cleanRegex=cleanRegex)
        
    def _createImgReader(self,imgPath,imgType):
        return rawReader(imgPath,imgType,self._image_index)

    def _createEvaluator(self,compID):
        if self.config['use_salt']: self._salt = str(uuid.uuid4())
        else: self._salt = None
        cachedir = os.path.join(self._path_root,\
                            'annotations_cache',\
                                self._image_set)
        if not osp.isdir(cachedir):
            os.makedirs(cachedir)
        return bboxEvaluator(self._datasetName,self.classes,
                             compID, "_" + self._salt,
                             cachedir, self._imageSetPath)

    def _update_image_index(self,newImageIndex):
        self._image_index = newImageIndex
        self.imgReader._image_index = newImageIndex     

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
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _load_classes(self, classFilename):
        with open(classFilename,"r") as f:
            classes = [line.rstrip() for line in f]
        return classes

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self.evaluator.evaluate_detections(all_boxes)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        print(cache_file)
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self.annoReader.load_annotation(index)
                    for index in self._image_index]

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True




