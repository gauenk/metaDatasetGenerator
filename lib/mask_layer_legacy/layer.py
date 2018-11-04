
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

ClsDataLayer implements a Caffe Python layer.
"""

import caffe
from utils.timer import Timer
from core.config import cfg
from datasets.ds_utils import convertFlattenedImageIndextoImageIndex
from aim_data_layer.minibatch import get_minibatch
import numpy as np
import numpy.random as npr
import yaml
from multiprocessing import Process, Queue

class MaskLayer(caffe.Layer):
    """
    Mask out values after pruning
    """

    def setNet(self,net):
        net.layer_dict[self.maskedLayer]
        
    def setup(self, bottom, top):
        """Setup the ClsDataLayer."""

        self.name = "MaskLayer"
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        self.maskedLayer = layer_params['maskedLayer']
        self.mask = []
        for idx in range(len(bottom)):
            self.mask.append(np.ones(bottom[idx].data.shape))
            print(bottom[idx].data.shape)
            top[idx].reshape(*bottom[idx].shape)
            print 'MaskLayer: mask shapes @ {}'.format(idx), self.mask[idx].shape
        print 'MaskLayer: mask length', len(self.mask)
        # self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        for idx in range(len(top)):
            top[idx].reshape(*(bottom[idx].shape))
            top[idx].data[...] = bottom[idx].data.astype(np.float32, copy=False) * self.mask[idx]


    def backward(self, top, propagate_down, bottom):
        """This layer is a window"""
        for idx in range(len(top)):
            bottom[idx].diff[...] = top[idx].diff.astype(np.float32, copy=False)

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

