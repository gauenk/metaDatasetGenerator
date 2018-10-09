
# --------------------------------------------------------
# CBAL
# Written by Kent Gauen
# --------------------------------------------------------

import os,json,sys
import os.path as osp
import numpy as np
import scipy.sparse
from core.config import cfg,cfgData
from easydict import EasyDict as edict
from datasets.ds_utils import xywh_to_xyxy

class jsonReader(object):
    """annoRead for reading json files."""

    def __init__(self, annoPath, classes , datasetName, setID, imageSet, bboxOffset = 0, useDiff = True, convertToPerson=None, convertIdToCls = None):
        """
        __init__ function for annoReader [annotationReader]

        """
        self._annoPath = annoPath
        self._bboxOffset = bboxOffset
        self._datasetName = datasetName
        self._setID = setID
        self._imageSet = imageSet
        self._convertToPerson = convertToPerson
        self.num_classes = len(classes)
        self.useDiff = useDiff
        self._classToIndex = self._create_classToIndex(classes)
        self._convertIdToCls = convertIdToCls

    def _create_classToIndex(self,classes):
        return dict(zip(classes, range(self.num_classes)))
        
    def load_annotation(self,index):
        """
        load annotations depending on how the annotation should be loaded

        """
        return self._load_json_annotation(index)

    def _getBaseImageSet(self):
        # return base name; "train", "test2014", "test2015", "val"
        if "train" in self._imageSet:
            return "train"
        elif "val" in self._imageSet:
            return "val2014"
        elif "test" in self._imageSet:
            return self._imageSet

    def _load_json_annotation(self, index):
        """
        requires the following format @ 
        <PATH_TO_ANNOTATIONS>/*xml
        """
        if type(index) is int:
            index = 'COCO_' + self._getBaseImageSet() + '_' +\
                        str(index).zfill(12)
        filename = os.path.join(self._annoPath, index + '.json')
        with open(filename,"r") as f:
            anno = json.load(f)

        # set the number of objects
        num_objs = 0
        for ix, obj in enumerate(anno['annotation']):
            cls = obj['category_id']
            if self._find_cls(cls) != -1:
                num_objs+=1

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        ix = 0
        for obj in anno['annotation']:
            cls = obj['category_id']
            cls = self._find_cls(cls)
            if cls == -1:
                continue
            boxes[ix, :] = xywh_to_xyxy(np.array(obj['bbox'])[np.newaxis,:])
            x1,y1,x2,y2 = boxes[ix, :]
            gt_classes[ix] = obj['category_id']
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
            ix += 1

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas,
                'set': self._setID}

    def _find_cls(self,cls):
        if type(cls) is int:
            return self._find_cls_int(cls)
        elif type(cls) is str:
            return self._find_cls_str(cls)
        else:
            print(type(cls))

    def _find_cls_int(self,cls):
        if cls < len(self._classToIndex):
            return cls
        else:
            return -1

    def _find_cls_str(self,cls):
        if self._convertIdToCls is not None: cls = self._convertIdToCls[cls]
        # check if we need to convert annotation class to "person"
        if self._convertToPerson is not None and cls in self._convertToPerson:
            cls = self._classToIndex["person"]
        # remove all not in classToIndex
        elif cls in self._classToIndex.keys():
            cls = self._classToIndex[cls]
        else:
            cls = -1
        return cls


    def _handle_caltech_helps_vs_gauenk(self,x1,y1,x2,y2):
        if "helps" in cfg.PATH_YMLDATASETS and "caltech" in cfgData.EXP_DATASET: 
            x1 = x1 * 640
            y1 = y1 * 480
            x2 = x2 * 640
            y2 = y2 * 480
        return x1,y1,x2,y2
        
