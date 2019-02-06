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
        self.minibatch_settings.siamese = self._siamese
        if self._siamese:
            self.minibatch_settings.load_fields = ['data_0','data_1','labels']
        else:
            self.minibatch_settings.load_fields = ['data','labels']
        self._shuffle_dataset_inds()

    def setup(self, bottom, top):
        """Setup the ClsDataLayer."""

        self.name = "ClsDataLayer"

        
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']
        self._siamese = False
        if 'siamese' in layer_params.keys():
            self._siamese = layer_params['siamese']

        self._generation = False
        if 'generation' in layer_params.keys():
            self._generation = layer_params['generation']
        self._name_to_top_map = {}

        idx = 0

        if self._siamese:
            top[idx].reshape(cfg.TRAIN.BATCH_SIZE, cfg.COLOR_CHANNEL,cfg.IMAGE_SIZE[0],cfg.IMAGE_SIZE[1])
            print("reshaped data_0")
            self._name_to_top_map['data_0'] = idx

            idx += 1
            top[idx].reshape(cfg.TRAIN.BATCH_SIZE, cfg.COLOR_CHANNEL,cfg.SIAMESE_IMAGE_SIZE[0],cfg.SIAMESE_IMAGE_SIZE[1])
            print("reshaped data_1")
            self._name_to_top_map['data_1'] = idx
        else:
            top[idx].reshape(cfg.TRAIN.BATCH_SIZE, cfg.COLOR_CHANNEL,cfg.IMAGE_SIZE[0],cfg.IMAGE_SIZE[1])
            print("reshaped data")
            self._name_to_top_map['data'] = idx

        idx += 1
        if self._generation:
            pass
            # top[idx].reshape(cfg.TRAIN.BATCH_SIZE,cfg.COLOR_CHANNEL,cfg.IMAGE_SIZE[0],cfg.IMAGE_SIZE[1])
        else:
            top[idx].reshape(cfg.TRAIN.BATCH_SIZE)
            print("reshaped labels")
            self._name_to_top_map['labels'] = idx


        print 'ClsDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)


    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        self._blobs = blobs
        #self.view_blobs(blobs)
        for blob_name, blob in blobs.iteritems():
            # print(blob_name,blob.shape)
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def view_blobs(self,blobs):
        if self._siamese:
            print("SIAMESE")
            from utils.blob import save_blob_list_to_file
            data_0 = blobs['data_0']
            data_1 = blobs['data_1']
            labels = np.squeeze(blobs['labels'])
            print("data0.shape",data_0.shape)
            print("data1.shape",data_1.shape)
            print("labels.shape",labels.shape)
            save_blob_list_to_file(data_0,labels,vis=False,size=cfg.CROPPED_IMAGE_SIZE,infix="0")
            save_blob_list_to_file(data_1,labels,vis=False,size=cfg.CROPPED_IMAGE_SIZE,infix="1")
        else:
            from utils.blob import save_blob_list_to_file
            data = blobs['data']
            labels = data # np.squeeze(blobs['labels'])
            labels_str = ["{}_{}".format(label,i) for i,label in enumerate(labels)]
            print("data.shape",data.shape)
            print("labels.shape",labels.shape)
            save_blob_list_to_file(data,labels_str,vis=False,size=cfg.CROPPED_IMAGE_SIZE)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

