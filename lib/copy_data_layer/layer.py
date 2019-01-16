"""The data layer used during training to train a Fast R-CNN network.

CopyDataLayer implements a Caffe Python layer.
"""

import caffe
from utils.timer import Timer
from core.config import cfg
from datasets.ds_utils import convertFlattenedImageIndextoImageIndex
from cls_data_layer.minibatch import get_minibatch
import numpy as np
import numpy.random as npr
import yaml
from multiprocessing import Process, Queue

class CopyDataLayer(caffe.Layer):
    """ Classification data layer used for training."""

    def set_data(self, input_data):
        self.input_data = input_data
        self.batch_size = cfg.TRAIN.BATCH_SIZE

    def setup(self, bottom, top):
        """Setup the CopyDataLayer."""

        self.name = "CopyDataLayer"

        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        idx = 0

        top[idx].reshape(cfg.TRAIN.BATCH_SIZE, 3,cfg.CROPPED_IMAGE_SIZE,cfg.CROPPED_IMAGE_SIZE)
        self._name_to_top_map['data'] = idx
        idx += 1
        print("reshaped data")

        top[idx].reshape(cfg.TRAIN.BATCH_SIZE)
        self._name_to_top_map['labels'] = idx
        idx += 1
        print("reshaped labels")

        print 'CopyDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)


    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        input_data = self.input_data
        for blob_name, blob in input_data.items():
            # print(blob_name,blob.shape)
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
        # top[0].reshape(self.batch_size, 3, cfg.TRAIN.MAX_SIZE,cfg.TRAIN.MAX_SIZE)
        # top[1].reshape(self.batch_size)

