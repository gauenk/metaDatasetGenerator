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

    """
    dataset_augmentation_bool_list = 
    - * - - * - - - * - -
    0 1 0 0 1 0 0 0 1 0 0

    0 1 6 7 8 13
      2     9
      3     10
      4     11
      5     12
    
    new_index_list =
    [0,1,6,7,8,13]
    
    ds_index # -1 is none
    image_index = 
    
    """

    def reset_dataset_augmentation_vars(self):
        roidb_size = len(self._roidb)
        num_roidb_aug = 0
        if cfg.DATASET_AUGMENTATION.BOOL: num_roidb_aug = int(roidb_size * self._perc_augmented)
        self.da_bool_array = np.zeros(roidb_size,dtype=np.int)
        # print(roidb_size,num_roidb_aug)

        if cfg.DATASET_AUGMENTATION.BOOL:
            plus_indices = npr.permutation(roidb_size)[:num_roidb_aug]
            self.da_bool_array[plus_indices] = 1
        self._num_augmented = num_roidb_aug * cfg.DATASET_AUGMENTATION.SIZE
        self._num_not_augmented = roidb_size - num_roidb_aug

        if cfg.DATASET_AUGMENTATION.BOOL: self.da_index_list = self.create_da_index_list()
        else: self.da_index_list = np.arange(self._num_samples,dtype=np.int)

    def create_da_index_list(self):
        index = 0
        roidb_size = len(self.da_bool_array)
        da_index_list = np.zeros(roidb_size,dtype=np.int)
        for sample_index,da_bool in enumerate(self.da_bool_array):
            da_index_list[sample_index] = index
            if da_bool: step = cfg.DATASET_AUGMENTATION.SIZE # LINK TO "OUTSIDE WORLD" [outside of this code block]
            else: step = 1
            index += step
        return da_index_list

    def get_indices_from_da_index(self,index):
        tmp_index = np.where(index < self.da_index_list)[0]
        if len(tmp_index) == 0: roidb_index = len(self.da_index_list) - 1
        else: roidb_index = tmp_index[0] - 1
        da_index = index - self.da_index_list[roidb_index]
        return roidb_index,da_index
        
    @property
    def _num_samples(self):
        # #num_samples = len(self._roidb)        
        # if cfg.DATASET_AUGMENTATION.BOOL:
        #     num_samples =  (self._roidb) * cfg.DATASET_AUGMENTATION.SIZE            
        num_samples = self._num_augmented + self._num_not_augmented
        # if cfg.DATASET_AUGMENTATION.BOOL:
        #     n_samples_aug = len(self.da_bool_array)
        #     n_samples_not_aug = len(self._roidb) - n_samples_aug
        #     num_samples =  n_samples_not_aug + n_samples_aug * cfg.DATASET_AUGMENTATION.SIZE
        return num_samples

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = npr.permutation(self._num_samples)
        self._cur = 0
        self.reset_dataset_augmentation_vars()

        # print(np.where(self._records == 0))
        # # print(self._roidb[np.where(self._records == 0)[0]][:10])
        # print(np.where(self._records == 0))
        # print(len(self._records))
        # print(np.sum(self._records == 0))
        # print(np.sum(self._records == 1))
        # print(np.sum(self._records))
        # print(self._records[:10])
        # sys.exit()

    def _balance_classes(self):
        pass

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        # print(cfg.TRAIN.BATCH_SIZE)
        # print(cfg.DATASET_AUGMENTATION.SIZE)
        # assert cfg.TRAIN.BATCH_SIZE >= cfg.DATASET_AUGMENTATION.SIZE, "size not geq"
        # assert (cfg.TRAIN.BATCH_SIZE % cfg.DATASET_AUGMENTATION.SIZE) == 0, "not divisible"
        num_samples = self._num_samples
        if self._cur + cfg.TRAIN.BATCH_SIZE >= num_samples:
            self._shuffle_roidb_inds()
        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.BATCH_SIZE]
        self._cur += cfg.TRAIN.BATCH_SIZE
        return db_inds

    def _get_roid_and_augmentation_list(self,db_inds):
        num_original_samples = len(self._roidb)
        roidb_index_list = []
        augmentation_index_list = []
        for ind in db_inds:
            roidb_index = ind
            da_index = 0
            if cfg.DATASET_AUGMENTATION.BOOL: roidb_index,da_index = self.get_indices_from_da_index(ind)
            roidb_index_list.append(roidb_index)
            augmentation_index_list.append(da_index)
        return roidb_index_list,augmentation_index_list

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()
            # ? is the "image_id" from a flattened imdb's roidb ?
            # ^ yes. 09/10/18
            # we assume records are *not flattened*
            roidb_inds,augmentation_inds = self._get_roid_and_augmentation_list(db_inds)
            minibatch_db = [self._roidb[i] for i in roidb_inds]
            da_bool_array = [self.da_bool_array[i] for i in roidb_inds]
            records_db = []
            if self._records is not None: records_db = [self._records[i] for i in db_inds]
            # records_db = self._extractRecordDb(minibatch_db)
            return get_minibatch(minibatch_db, records_db, self._num_classes, augmentation_inds,da_bool_array)

    def _extractRecordDb(self,minibatch_db):
        records_db = []
        for roi in minibatch_db:
            image_index,bbox_index = convertFlattenedImageIndextoImageIndex(roi['image_id'])
            if cfg.DATASET_AUGMENTATION.BOOL:
                for repeat in range(cfg.DATASET_AUGMENTATION.SIZE): # for dataset augmentation
                    records_db.append(self._records[image_index][bbox_index])
        return records_db

    def set_roidb(self, roidb, records, perc_augmented=0):
        """Set the roidb to be used by this layer during training."""
        assert 'image_id' in roidb[0].keys()
        self._roidb = roidb
        self._records = records
        self._perc_augmented = perc_augmented
        #self._num_samples = len(roidb)

        # dataset augmentation
        self.reset_dataset_augmentation_vars()

        if cfg.TRAIN.CLS.BALANCE_CLASSES and records is not None:
            # assume order is preserved in "extractRecordDb"
            print("TODO: records has some issues.")
            print("FIX THIS for DATASET AUGMENTATION!!")
            self._records = np.array(self._extractRecordDb(self._roidb))
            neg = np.sum(self._records == 0)
            pos = np.sum(self._records == 1)

            if pos >= .9*neg and pos <= 1.1*neg: # a 10% class imbalance is okay
                self._perm = npr.permutation(num_samples)
            elif pos >= .9*neg: # (given we failed the 1st cond.) the positives are too big
                # we grow the negatives to 110%|#positives|
                print("[set_roidb: balancing classes]: growing the negatives")
                goal = 1.1*pos
                neg_inds = np.where(self._records == 0)[0]
                neg_inds_rand = npr.permutation(neg_inds)[:goal] # limit to only needed
                neg_roidb_examples = [ self._roidb[idx] for idx in neg_inds ]
                neg_record_examples = [ self._records[idx] for idx in neg_inds ]
                self._roidb.extend(neg_roidb_examples)
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

        self.name = "ClsDataLayer"
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        # data blob: holds a batch of N samples, each with 3 channels
        idx = 0

        top[idx].reshape(cfg.TRAIN.BATCH_SIZE, 3,
                         cfg.CROPPED_IMAGE_SIZE,cfg.CROPPED_IMAGE_SIZE)
        self._name_to_top_map['data'] = idx
        idx += 1
        print("reshaped data")

        top[idx].reshape(cfg.TRAIN.BATCH_SIZE)
        self._name_to_top_map['labels'] = idx
        idx += 1
        print("reshaped labels")

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
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

