
# --------------------------------------------------------
# CBAL
# Written by Kent Gauen
# --------------------------------------------------------

import os
import os.path as osp
import numpy as np
import scipy.sparse
from core.config import cfg,cfgData
from easydict import EasyDict as edict
import xml.etree.ElementTree as ET


class xmlReader(object):
    """Image database."""

    def __init__(self, annoPath, classes , datasetName, setID, bboxOffset = 0, useDiff = True, convertToPerson=None, convertIdToCls = None):
        """
        __init__ function for annoReader [annotationReader]

        """
        self._annoPath = annoPath
        self._bboxOffset = bboxOffset
        self._datasetName = datasetName
        self._setID = setID
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
        return self._load_xml_annotation(index)

    def _load_xml_annotation(self, index):
        """
        requires the following format @ 
        <PATH_TO_ANNOTATIONS>/*xml
        """
        filename = os.path.join(self._annoPath, index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.useDiff:
            # Exclude the samples labeled as difficult
            objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]

        # set the number of objects
        num_objs = 0
        for ix, obj in enumerate(objs):
            if self._find_cls(obj) != -1:
                num_objs+=1
                
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        ix = 0
        for obj in objs:
            bbox = obj.find('bndbox')
            cls = self._find_cls(obj)
            if cls == -1:
                continue
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - self._bboxOffset
            y1 = float(bbox.find('ymin').text) - self._bboxOffset
            x2 = float(bbox.find('xmax').text) - self._bboxOffset
            y2 = float(bbox.find('ymax').text) - self._bboxOffset
            # handle scaling caltech; annos were modified for YOLO
            x1,y1,x2,y2 = self._handle_caltech_helps_vs_gauenk(*[x1, y1, x2, y2])
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
            ix += 1

        overlaps = scipy.sparse.csr_matrix(overlaps)
        # TODO: convert the 'set' into an index for a vector;
        # add the conversion of the layer in the "train.prototxt"
        # of the VAE 
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas,
                'set': self._setID}
                #'set': self._datasetName}

    def _find_cls(self,obj):
        cls = obj.find('name').text.lower().strip()
        #TODO: sun. "person model" is not accounted for (missing 17 annos)
        # find out what is happening to them

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
        
