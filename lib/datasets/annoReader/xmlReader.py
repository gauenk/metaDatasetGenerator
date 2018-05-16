
# --------------------------------------------------------
# CBAL
# Written by Kent Gauen
# --------------------------------------------------------

import os
import os.path as osp
import PIL
import numpy as np
import scipy.sparse
from core.config import cfg
from easydict import EasyDict as edict
import xml.etree.ElementTree as ET


class xmlReader(object):
    """Image database."""

    def __init__(self, annoPath, classes , datasetName, bboxOffset = 0, useDiff = True, convertToPerson=False):
        """
        __init__ function for annoReader [annotationReader]

        """
        self._annoPath = annoPath
        self._bboxOffset = bboxOffset
        self._datasetName = datasetName
        self.num_classes = len(classes)
        self.useDiff = useDiff
        self._classToIndex = self._create_classToIndex(classes)

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
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - self._bboxOffset
            y1 = float(bbox.find('ymin').text) - self._bboxOffset
            x2 = float(bbox.find('xmax').text) - self._bboxOffset
            y2 = float(bbox.find('ymax').text) - self._bboxOffset
            cls = self._classToIndex[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)
        # TODO: convert the 'set' into an index for a vector;
        # add the conversion of the layer in the "train.prototxt"
        # of the VAE 
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas,
                'set': 1}
                #'set': self._datasetName}
