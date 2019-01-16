"""The data layer used during training to train a Fast R-CNN network.

ClsDataLayer implements a Caffe Python layer.
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

class ClsDataLayer(caffe.Layer):
    """ Classification data layer used for training."""

    @property
    def _num_samples(self):
        return self.num_samples

    def _shuffle_dataset_inds(self):
        """Randomly permute the training sample indices."""
        self._perm = npr.permutation(self.num_samples)
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the samples indices for the next minibatch."""
        num_samples = self._num_samples
        if self._cur + cfg.TRAIN.BATCH_SIZE >= num_samples:
            self._shuffle_dataset_inds()
        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.BATCH_SIZE]
        self._cur += cfg.TRAIN.BATCH_SIZE
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch."""
        minibatch_indices = self._get_next_minibatch_inds()
        minibatch_samples,minibatch_scales = self.ds_loader.minibatch(minibatch_indices,self.minibatch_settings,load_as_blob=True)
        return minibatch_samples

    def set_data_loader(self, ds_loader, minibatch_settings):
        self.ds_loader = ds_loader
        self.num_samples = ds_loader.num_samples
        self.minibatch_settings = minibatch_settings
        self.minibatch_settings.load_fields = ['data','labels']
        self._shuffle_dataset_inds()

    def setup(self, bottom, top):
        """Setup the ClsDataLayer."""

        self.name = "ClsDataLayer"

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

        print 'ClsDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)


    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        self._blobs = blobs
        for blob_name, blob in blobs.iteritems():
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

