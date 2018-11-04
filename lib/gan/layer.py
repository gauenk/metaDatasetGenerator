
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
from cls_data_layer.minibatch import get_minibatch
import numpy as np
import numpy.random as npr
import yaml
from multiprocessing import Process, Queue

class GAN_JOINT(caffe.Layer):
    """GAN layer used for combining the generator and discriminator."""

    def _extractRecordDb(self,minibatch_db):
        records_db = []
        for roi in minibatch_db:
            image_index,bbox_index = convertFlattenedImageIndextoImageIndex(roi['image_id'])
            records_db.append(self._records[image_index][bbox_index])
        return records_db

    def set_roidb(self, roidb, records):
        """Set the roidb to be used by this layer during training."""
        assert 'image_id' in roidb[0].keys()
        self._roidb = roidb
        self._records = records
        if cfg.TRAIN.CLS.BALANCE_CLASSES:
            # assume order is preserved in "extractRecordDb"
            self._records = np.array(self._extractRecordDb(self._roidb))
            neg = np.sum(self._records == 0)
            pos = np.sum(self._records == 1)

            if pos >= .9*neg and pos <= 1.1*neg: # a 10% class imbalance is okay
                self._perm = npr.permutation(np.arange(len(self._roidb)))
            elif pos >= .9*neg: # (given we failed the 1st cond.) the positives are too big
                # we grow the negatives to 110%|#positives|
                goal = 1.1*pos
                neg_inds = np.where(self._records == 0)[0]
                neg_inds_rand = npr.permutation(neg_inds)[:goal] # limit to only needed
                neg_roidb_examples = [ self._roidb[idx] for idx in neg_inds ]
                neg_record_examples = [ self._records[idx] for idx in neg_inds ]
                print(self._roidb[-2],self._roidb[-1])
                self._roidb.extend(neg_roidb_examples)
                print(self._roidb[-300],self._roidb[-2],self._roidb[-1])
                self._records = np.r_[self._records,neg_record_examples]
            elif pos <= 1.1*neg: # (given we failed the 1st cond.) the positives are too small
                goal = neg - pos
                pass # not here yet

        self._shuffle_roidb_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(10)
            self._prefetch_process = BlobFetcher(self._blob_queue,
                                                 self._roidb,
                                                 self._num_classes)
            self._prefetch_process.start()
            # Terminate the child process when the parent exists
            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()
            import atexit
            atexit.register(cleanup)

    def setup(self, bottom, top):
        """Setup the ClsDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0

        top[idx].reshape(cfg.TRAIN.BATCH_SIZE, 3,
                         cfg.CROPPED_IMAGE_SIZE,cfg.CROPPED_IMAGE_SIZE)
        self._name_to_top_map['data'] = idx
        idx += 1

        top[idx].reshape(cfg.TRAIN.BATCH_SIZE)
        self._name_to_top_map['labels'] = idx
        idx += 1

        # if cfg.TRAIN.OBJ_DET.HAS_RPN:
        #     top[idx].reshape(1)
        #     self._name_to_top_map['im_info'] = idx
        #     idx += 1


        print 'ClsDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)




    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        # timer = Timer()
        # timer.tic()
        blobs = self._get_next_minibatch()
        self._blobs = blobs
        # print(timer.toc())
        #print(blobs['data'].shape)

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """
        We need to propogate backward through both models
        """
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

