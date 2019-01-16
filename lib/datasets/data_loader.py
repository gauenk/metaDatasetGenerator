import cv2,copy
import numpy as np
import numpy.random as npr
from easydict import EasyDict as edict
from utils.blob import preprocess_image_for_model,im_list_to_blob
from datasets.data_utils.dataset_augmentation_utils import applyDatasetAugmentation

class DataLoader():

    def __init__(self,imdb,records,dataset_augmentation):
        self.imdb = imdb
        self.roidb = imdb.roidb
        self.image_index = imdb.image_index
        self.roidb_size = len(self.roidb)
        self.correctness_records = records
        self.dataset_augmentation = edict()
        self.dataset_augmentation.any_augmented = dataset_augmentation.BOOL
        self.dataset_augmentation.dataset_percent = dataset_augmentation.N_SAMPLES
        self.dataset_augmentation.size = dataset_augmentation.SIZE
        self.dataset_augmentation.configs = dataset_augmentation.CONFIGS
        self.num_samples = self.get_dataset_size(self.roidb_size)
        self.dataset_augmentation.sample_bools = self.get_sample_augmentation_bools()
        self.dataset_augmentation.augmentation_indices = self.get_augmentation_indices()

    def __len__(self):
        return self.num_samples
    
    @property
    def size(self):
        return self.num_samples

    def get_dataset_size(self,num_original_samples):
        if not self.dataset_augmentation.any_augmented:
            return num_original_samples
        num_samples_to_augment = int(num_original_samples * self.dataset_augmentation.dataset_percent)
        num_augmented_samples = num_samples_to_augment * self.dataset_augmentation.size
        num_samples_no_augment = num_original_samples - num_samples_to_augment
        num_samples = num_augmented_samples + num_samples_no_augment
        self.num_samples_to_augment = num_samples_to_augment
        self.num_samples_no_augment = num_samples_no_augment
        return num_samples

    def get_sample_augmentation_bools(self):
        if not self.dataset_augmentation.any_augmented:
            return np.zeros(self.roidb_size,dtype=np.bool)
        sample_bools = np.zeros(self.roidb_size,dtype=np.int)
        augmented_sample_indices = npr.permutation(self.roidb_size)[:self.num_samples_to_augment]
        sample_bools[augmented_sample_indices] = 1
        return sample_bools
        
    def get_augmentation_indices(self):
        if not self.dataset_augmentation.any_augmented:
            return np.zeros(self.roidb_size,dtype=np.bool)
        index = 0
        augmentation_indices = np.zeros(self.roidb_size,dtype=np.int)
        for sample_index,aug_bool in enumerate(self.dataset_augmentation.sample_bools):
            augmentation_indices[sample_index] = index
            if aug_bool:
                step = self.dataset_augmentation.size
            else:
                step = 1
            index += step
        return augmentation_indices

    def convert_index(self,index):
        # check if no sample augmented
        if not self.dataset_augmentation.any_augmented:
            return index,None
        roidb_index = self.get_roidb_index(index)
        # check if this sample is augmented
        if not self.dataset_augmentation.sample_bools[roidb_index]:
            return roidb_index,0
        aug_index = self.get_aug_index(index,roidb_index)
        return roidb_index,aug_index

    def get_roidb_index(self,index):
        tmp_index = np.where(index < self.dataset_augmentation.augmentation_indices)[0]
        if len(tmp_index) == 0:
            roidb_index = self.roidb_size - 1
        else:
            roidb_index = tmp_index[0] - 1
        return roidb_index

    def get_aug_index(self,index,roidb_index):
        aug_index = index - self.dataset_augmentation.augmentation_indices[roidb_index]
        return aug_index
        
    def add_dataset_augmentation_information(self,index):
        roidb_index,aug_index = self.convert_index(index)
        sample = self.roidb[roidb_index]
        sample['aug_bool'] = self.dataset_augmentation.sample_bools[roidb_index]
        sample['aug_index'] = aug_index
        return sample,roidb_index
        
    def add_records_information(self,sample,roidb_index):
        if self.correctness_records:
            sample['correctness_record'] = self.correctness_records[roidb_index]        
            sample['correctness_bool'] = True
        else:
            sample['correctness_record'] = None
            sample['correctness_bool'] = False

    def add_image_id_information(self,sample,roidb_index):
        sample['image_id'] = self.image_index[roidb_index]
        sample['index'] = roidb_index
        
    def get_sample_with_info(self,index):
        sample,roidb_index = self.add_dataset_augmentation_information(index)
        self.add_records_information(sample,roidb_index)
        self.add_image_id_information(sample,roidb_index)
        return sample

    def balance_classes(self):
        raise NotImplemented("we can not balance classes currently")
    #
    # load sample from sample list with information
    #

    def sample_minibatch_roidbs(self,indices):
        """
        proposed alternative to returning only roidbs
        -> records stored in returned roibs
        -> augmentation_bool
        -> augmentation_index
        """
        sampleList = []
        for index in indices:
            sample = self.get_sample_with_info(index)
            sampleList.append(sample)
        return sampleList
            
    def sample_minibatch_roidbs_generator(self,indices):
        """
        proposed alternative to returning only roidbs
        -> records stored in returned roibs
        -> augmentation_bool
        -> augmentation_index
        """
        for index in indices:
            sample = self.get_sample_with_info(index)
            yield sample

    #
    # loading the sample information into a useable format (e.g. load image to ndarray)
    #

    def minibatch(self,indices,load_settings,load_as_blob=False,load_image=True):
        """
        note load_as_blob does this *per batch*
        """
        imageList,scaleList,labelList = [],[],[]
        # prepare the settings per sample
        load_settings_for_sample = copy.deepcopy(load_settings)
        load_settings_for_sample.load_fields = []
        for sample in self.sample_minibatch_roidbs_generator(indices):
            image,scale = self.load_sample(sample,load_settings_for_sample,False,load_image) # always false since we want to return it all as *one* blob, not multiple
            imageList.append(image)
            scaleList.append(scale)
            labelList.append(sample['gt_classes'])
        if load_as_blob:
            imageList = im_list_to_blob(imageList) # returns blob
        return self.format_return_values(imageList,labelList,scaleList,load_settings)

    def minibatch_generator(self,indices,loadConfig,load_as_blob=False,load_image=True):
        """
        written with testing in mind;
        note load_as_blob does this *per image*
        """
        dataset_index = 0
        for sample in self.sample_minibatch_roidbs_generator(indices):
            image,scale = self.load_sample(sample,loadConfig,load_as_blob,load_image)
            yield image,scale,sample,dataset_index
            dataset_index += 1

    def dataset_generator(self,loadConfig,load_as_blob=False,load_image=True):
        indicies = np.arange(self.num_samples)
        return self.minibatch_generator(indicies,loadConfig,load_as_blob,load_image)

    #
    # primary loading functions
    #

    def load_sample(self,sample,load_settings,load_as_blob=False,load_image=True):
        if load_image is False:
            return None,None
        image_path = self.imdb.image_path_at(sample['image_id'])
        img = cv2.imread(image_path)
        scales = []
        if load_settings.cropped_to_box_bool:
            img = cropImageToAnnoRegion(img,sample['boxes'][load_settings.cropped_to_box_index]) 
        if sample['aug_bool']:
            transforms = self.dataset_augmentation.configs[sample['aug_index']]
            inputTransDict = {'transformations':transforms}
            img = applyDatasetAugmentation(img,inputTransDict)
        if sample['correctness_bool']:
            raise ValueError("unknown handling of records")
        if load_settings.load_rois_bool:
            raise ValueError("unknown handling of rois")
        if load_settings.activation_sample.bool_value:
            raise ValueError("unknown handling of activation values")
        if load_settings.preprocess_image:
            img, scales = preprocess_image_for_model(img, load_settings.dataset_means,load_settings.target_size,load_settings.max_sample_single_dimension_size)            
        if load_as_blob:
            img = im_list_to_blob([img]) # returns blob
        label = sample['gt_classes']
        return self.format_return_values(img,label,scales,load_settings)

    def format_return_values(self,imgs,labels,scales,load_settings):
        # return options
        if len(load_settings.load_fields) == 0:
            return imgs, scales
        else:
            returnDict = {}
            for load_field in load_settings.load_fields:
                if 'data' == load_field :
                    returnDict['data'] = np.array(imgs)
                elif 'labels' == load_field:
                    returnDict['labels'] = np.array(labels)
                elif 'im_info':
                    returnDict['labels'] = None
                elif 'records' == load_field:
                    returnDict['records'] = np.array([])
            return returnDict,scales

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

