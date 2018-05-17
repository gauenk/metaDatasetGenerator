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
from glob
from core.config import cfg,get_output_dir, createFilenameID
from easydict import EasyDict as edict
from datasets.evaluators.bboxEvaluator import bboxEvaluator

class SpoofImdb(imdb):
    """Spoof an IMDB to run the testing script for mix data."""

    def __init__(self, datasetName, configName):
        imdb.__init__(self, datasetName)
        """
        datasetName = 8 bit vector to indicate which repo datasets are mixed
        configName = configuration file to load
        """
        self._local_path = os.path.dirname(__file__)
        cfg.EXP_DIR = datasetName
        self._classes = ["__background__","person"]
        self._datasetName = datasetName
        self._configName = configName
        self._roidb = None
        self._roidb_handler = self.gt_roidb
        self._roidbSize = []
        self._convertIdToCls = None
        self._setupConfig()
        self.evaluator = self._createEvaluator("meaninglessCompID")

    def load_roidb(self, repeat, size):
        """
        repeat = which repeat to load
        initSize: dataset size to load
        """
        sizeIndexEnd = cfg.mixtureSizes.index(size)
        for mixureSize in cfg.mixtureSizes[:sizeIndexEnd]:
            extRoidb = self._load_mixture_roidb(repeat,mixtureSize)
            self._roidb.extend(extRoidb)
        return self._roidb

    def _load_mixture_roidb(self,repeat,size):
        """
        reads in the 
        """
        mixtureFile = createFilenameID(self.name,self.r,size)
        if osp.exists(mixtureFile) == False:
            raise KeyError("mixture {} doesn't exist.".format(mixtureFile))
        with open(mixtureFile, 'rb') as fid:
            roidb = pickle.load(fid)
        print('{} gt roidb loaded from {}'.format(self.name, mixtureFile))
        return roidb

    def image_path_at(self, i):
        if self._roidb is None:
            raise ValueError("SpoofImdb's image_path_at requires the roidb be loaded.")
        return self._roidb[i]['image']
        
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
        if self._roidb is None:
            raise ValueError("SpoofImdb's gt_roidb requires .load_roidb(repeat,size) be called first")
        return self._roidb

    def _setupConfig(self):
        fn = osp.join(self._local_path,"ymlConfigs" ,self._configName + ".yml")
        with open(fn, 'r') as f:
            yaml_cfg = edict(yaml.load(f))
        fn = osp.join(self._local_path,"ymlConfigs" ,yaml_cfg['CONFIG_DATASET_INDEX_DICTIONARY'])
        with open(fn, 'r') as f:
            setID = edict(yaml.load(f))[self._datasetName]
        self.config = {'cleanup'     : yaml_cfg['CONFIG_CLEANUP'],
                       'use_salt'    : yaml_cfg['CONFIG_USE_SALT'],
                       'use_diff'    : yaml_cfg['CONFIG_USE_DIFFICULT'],
                       'rpn_file'    : yaml_cfg['CONFIG_RPN_FILE'],
                       'min_size'    : yaml_cfg['CONFIG_MIN_SIZE'],
                       'setID'       : setID}
        
    def compute_size_along_roidb(self):
        if self.roidb is None:
            raise ValueError("roidb must be loaded before 'compute_size_along_roidb' can be run")
        self._roidbSize.append(len(self.roidb[0]['boxes']))
        for image in self.roidb[1:]:
            newSize = self._roidbSize[-1] + len(image['boxes'])
            self._roidbSize.append(newSize)
