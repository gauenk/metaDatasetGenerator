import cv2,copy
import numpy as np
import numpy.random as npr
from easydict import EasyDict as edict
from utils.base import readNdarray
from utils.blob import preprocess_image_for_model,im_list_to_blob,siameseImagesToBlobs
from datasets.data_utils.dataset_augmentation_utils import applyDatasetAugmentation,rotateImage
from datasets.data_utils.sample_loader import loadSampleDataWithEdictSettings,loadAdditionalInput
from core.test_utils.agg_activations import aggregateActivations
from utils.image_utils import splitImageForSiameseNet
from cache.roidb_cache import RoidbCache

class DataLoader():

    def __init__(self,imdb,records,dataset_augmentation,transform_each_sample,roidb_cache):
        self.imdb = imdb
        self.roidb = imdb.roidb
        self.image_index = imdb.image_index
        self.roidb_size = len(self.roidb)
        self.roidb_cache = roidb_cache
        self.roidb_cache_fieldname_prefix = 'data_loader'
        self.correctness_records = records
        self.dataset_augmentation = edict()
        self.set_dataset_augmentation(dataset_augmentation)
        self.num_samples = self.get_dataset_size(self.roidb_size)
        self.dataset_augmentation.sample_bools = self.get_sample_augmentation_bools()
        self.dataset_augmentation.augmentation_indices = self.get_augmentation_indices()
        self.set_transform_each_sample(transform_each_sample)
        self.replace_labels_after_augmentation = None
        self.rotate_index = None

    def __len__(self):
        return self.num_samples
    
    @property
    def size(self):
        return self.num_samples

    def set_dataset_augmentation(self,dataset_augmentation):
        self.dataset_augmentation.any_augmented = dataset_augmentation.BOOL
        self.dataset_augmentation.dataset_percent = dataset_augmentation.N_SAMPLES
        self.dataset_augmentation.size = dataset_augmentation.SIZE
        self.dataset_augmentation.configs = dataset_augmentation.CONFIGS
        
    def set_transform_each_sample(self,transform_each_sample):
        self.transform_each_sample = edict()

        self.transform_each_sample.data = edict()
        self.transform_each_sample.data.bool = transform_each_sample.DATA.BOOL
        self.transform_each_sample.data.rand = transform_each_sample.DATA.RAND
        self.transform_each_sample.data.type = transform_each_sample.DATA.TYPE
        self.transform_each_sample.data.type_params = transform_each_sample.DATA.TYPE_PARAMS

        self.transform_each_sample.label = edict()
        self.transform_each_sample.label.bool = transform_each_sample.LABEL.BOOL
        self.transform_each_sample.label.rand = transform_each_sample.LABEL.RAND
        self.transform_each_sample.label.type = transform_each_sample.LABEL.TYPE
        self.transform_each_sample.label.type_params = transform_each_sample.LABEL.TYPE_PARAMS
        
        # init some variables based on transform types.
        if self.transform_each_sample.label.bool and self.transform_each_sample.label.type == "file_replace":
            filename = self.transform_each_sample.label.type_params['file_replace']['filename']
            new_labels = readNdarray(filename)
            self.transform_each_sample.label.type_params['file_replace']['labels'] = new_labels
            assert self.num_samples == len(new_labels), "the replacing labels size needs to be equal"

        assert self.transform_each_sample.data.type in self.transform_each_sample.data.type_params.keys(), "we must have the type [{}] parameters in the data.type_param".format(self.transform_each_sample.type)
        assert self.transform_each_sample.label.type in self.transform_each_sample.label.type_params.keys(), "we must have the type [{}] parameters in the label.type_param".format(self.transform_each_sample.type)
        
        self.set_transform_each_sample_info()

    def get_dataset_size(self,num_original_samples):
        self.num_original_samples = num_original_samples
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
        augment_postfix_name = 'augmented_sample_indices_{:d}'.format(self.num_samples_to_augment)
        roidb_cache_fieldname = "{}_{}".format(self.roidb_cache_fieldname_prefix,augment_postfix_name)
        augmented_sample_indices = self.roidb_cache.load(roidb_cache_fieldname)
        if augmented_sample_indices is None:
            augmented_sample_indices = npr.permutation(self.roidb_size)[:self.num_samples_to_augment]
            self.roidb_cache.save(roidb_cache_fieldname,augmented_sample_indices)
        sample_bools[augmented_sample_indices] = 1
        return sample_bools
        
    def get_augmentation_indices(self):
        """
        0    1    2    3    4    5    (roidb index)
        0    4    5    9    13   14   (dataset augmentation)
        """
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

    def set_transform_each_sample_info(self):
        self.get_transform_each_sample_info_data()

    def get_transform_each_sample_info_data(self):
        """
        1. finish the transform_info
        2. re-align the "config" for [dataset augmentation] and [rotate image]
            -> they should not both be truth (i don't think(
            -> 
        """
        if self.transform_each_sample.data.bool is False:
            return np.zeros(self.roidb_size,dtype=np.bool)
        if self.transform_each_sample.data.rand:
            transform_cache_postfix = "{}_index".format(self.transform_each_sample.data.type)
            roidb_cache_fieldname = "{}_{}".format(self.roidb_cache_fieldname_prefix,transform_cache_postfix)
            self.transform_each_sample.data.roidb_info = self.roidb_cache.load(roidb_cache_fieldname)
            if self.transform_each_sample.data.roidb_info is None:
                index = self.get_transform_each_sample_info_by_type(self.transform_each_sample.data.type,self.transform_each_sample.data.type_params)
                self.transform_each_sample.data.roidb_info = index
                self.roidb_cache.save(roidb_cache_fieldname,index)
            return self.transform_each_sample.data.roidb_info
        else:
            print("[data_loader] transform each sample is not random... lame. Quitting.")
            exit()
        
        
    def get_transform_each_sample_info_by_type(self,transform_type,transform_params_dict):
        transform_params = transform_params_dict[transform_type]
        if transform_type == 'rotate':
            angle_min = transform_params['angle_min']
            angle_max = transform_params['angle_max']
            rotate_index = (npr.rand(self.roidb_size) * 360).astype(np.int)
            return rotate_index
        else:
            print("[data_loader] unknown transform type: {}. quitting.".format(transform_type))
            exit()

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
        sample['roidb_index'] = roidb_index
        sample['aug_bool'] = self.dataset_augmentation.sample_bools[roidb_index]
        if sample['aug_bool']:
            sample['aug_index'] = aug_index
        else:
            sample['aug_index'] = -1
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
        
    def add_sample_transform_information(self,sample,roidb_index,index):
        # add info for transforming data
        if self.transform_each_sample.data.bool:
            sample['data_transform_info'] = self.transform_each_sample.data.roidb_info[roidb_index]
        # complete the modification for the labels here
        if self.transform_each_sample.label.bool:
            if self.transform_each_sample.label.type == "file_replace":
                new_labels = self.transform_each_sample.label.type_params['file_replace']['labels']
                index_type = self.transform_each_sample.label.type_params['file_replace']['index_type']
                if index_type == 'roidb_index':
                    sample['gt_classes'] = new_labels[roidb_index]
                elif index_type == 'index':
                    sample['gt_classes'] = new_labels[index]
                else:
                    print("unkown label paraam index type {}".format(params.index_type))
                # sample['label_transform_info'] = self.transform_each_sample.label.roidb_info[roidb_index]

    def replace_label(self,sample,index):
        if self.replace_labels_after_augmentation is not None:
            # print("[data_loader.py: replace_label] replacing labels with ones from file")
            # print("\n\n\n\n\n\n\n\n\n\n\n\n\n")
            # print("REMOVE ME I MODIFIED THE OUTPUT TO ZERO ONE")
            # print("\n\n\n\n\n\n\n\n\n\n\n\n\n")
            sample['gt_classes'] = self.replace_labels_after_augmentation[index]
            #sample['gt_classes'] = (1.+self.replace_labels_after_augmentation[index])/2.
            
    def get_sample_with_info(self,index):
        sample,roidb_index = self.add_dataset_augmentation_information(index)
        self.add_records_information(sample,roidb_index)
        self.add_image_id_information(sample,roidb_index)
        self.add_sample_transform_information(sample,roidb_index)
        #self.replace_label(sample,index)
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
            label = sample['gt_classes']
            labelList.append(label)
        if load_as_blob:
            if load_settings.siamese:
                imageList = siameseImagesToBlobs(imageList)
            else:
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

    def set_dataset_loader_config(self,dataset_loader_config):
        self.dataset_loader_config = dataset_loader_config
        if self.dataset_loader_config.additional_input.bool:
            if self.dataset_loader_config.additional_input.type == "activations":
                from datasets.factory import loadImdbFromConfig
                activation_settings = self.dataset_loader_config.additional_input.info["activations"]
                cfg = activation_settings.cfg
                activations_cfg = activation_settings.activations_cfg
                agg_imdb = loadImdbFromConfig(cfg)
                activation_settings.agg_imdb = agg_imdb
                activation_settings.agg = aggregateActivations(activations_cfg,cfg,agg_imdb)
                activation_settings.agg.load(activations_cfg.LAYER_NAMES)

    def dataset_generator(self,loadConfig=None,load_as_blob=False,load_image=True):
        if loadConfig is not None:
            self.set_dataset_loader_config(loadConfig)
        if self.dataset_loader_config is None:
            print("[data_loader.py] ERROR: no data_loader_config for dataset_generator.")
        indicies = np.arange(self.num_samples)
        #print(self.dataset_loader_config)
        return self.minibatch_generator(indicies,self.dataset_loader_config,load_as_blob,load_image)

    def apply_sample_transformation(self,sample,img):
        trans_img = img
        if self.transform_each_sample.data.bool and img is not None:
            if self.transform_each_sample.type == 'rotate':
                angle = sample['data_transform_info']
                trans_img,_ = rotateImage(img,angle)
            else:
                print("[data_loader] unknown transformation data.type of each sample [{}]. quitting.".format(self.transform_each_sample.data.type))
                exit()
        if self.transform_each_sample.label.bool and not self.transform_each_sample.label.apply_at_get_sample_with_info_bool:
            #TODO; this code doesn't work
            print("no this is bad. don't go here.")
            exit()
            if self.transform_each_sample.type == 'file':
                new_label = ['label_transform_info']
                self.replace_labels_after_augmentation
                
                trans_img,_ = rotateImage(img,angle)
            else:
                print("[data_loader] unknown transformation data.type of each sample [{}]. quitting.".format(self.transform_each_sample.data.type))
                exit()

        return trans_img

    #
    # primary loading functions
    #

    def load_sample(self,sample,load_settings,load_as_blob=False,load_image=True):
        #self.modify_label(sample)
        if load_image is False:
            return None,None
        label = sample['gt_classes']
        image_path = self.imdb.image_path_at(sample['image_id'])
        img = cv2.imread(image_path)
        scales = []
        if load_settings.cropped_to_box_bool:
            img = cropImageToAnnoRegion(img,sample['boxes'][load_settings.cropped_to_box_index]) 
        if self.transform_each_sample.bool:
            img = self.apply_sample_transformation(sample,img)
        if sample['aug_bool']:
            assert sample['aug_index'] >= 0, "if the sample is augmented the aug_index must be non-negative"
            transforms = self.dataset_augmentation.configs[sample['aug_index']]
            inputTransDict = {'transformations':transforms}
            img = applyDatasetAugmentation(img,inputTransDict)
        if sample['correctness_bool']:
            raise ValueError("unknown handling of records")
        if load_settings.load_rois_bool:
            raise ValueError("unknown handling of rois")
        if load_settings.activation_sample.bool_value:
            raise ValueError("unknown handling of activation values")
        img = loadAdditionalInput(load_settings,img,sample,self.imdb)
        if load_settings.preprocess_image:
            if load_settings.siamese:
                img, scales = preprocess_image_for_model(img, load_settings.dataset_means,load_settings.target_size_siamese,load_settings.max_sample_single_dimension_size)
            else:
                img, scales = preprocess_image_for_model(img, load_settings.dataset_means,load_settings.target_size,load_settings.max_sample_single_dimension_size)                
        # we no longer concat images in the "loadAdditional" function 
        # if load_settings.siamese:
        #     img = splitImageForSiameseNet(img) 
        if load_as_blob:
            if load_settings.siamese:            
                img = siameseImagesToBlobs([img])
            else:
                img = im_list_to_blob([img]) # returns blob
        return self.format_return_values(img,label,scales,load_settings)

    def modify_label(self,sample):
        # GAH> this should merge with when we load labels.. can't we just modify the labels when we run "load_sample"? not in the "apply_at_the_..." ugh. that is gross.
        transforms = self.dataset_augmentation.configs[sample['aug_index']]
        sample['gt_classes'] = transforms[2]['angle'] / 360.
        # label = sample['gt_classes']        

    def format_return_values(self,imgs,labels,scales,load_settings):
        """
        ? maybe this functionality is better suited in the "get_sample_info" function?
        """
        # return options
        if len(load_settings.load_fields) == 0:
            return imgs, scales
        else:
            returnDict = {}
            for load_field in load_settings.load_fields:
                if 'data' == load_field:
                    returnDict['data'] = np.array(imgs)
                elif 'labels' == load_field:
                    returnDict['labels'] = np.array(labels)
                elif 'im_info' == load_field:
                    returnDict['labels'] = None
                elif 'records' == load_field:
                    returnDict['records'] = np.array([])
                elif 'data_0' == load_field:
                    returnDict['data_0'] = np.array(imgs[0])
                elif 'data_1' == load_field:
                    returnDict['data_1'] = np.array(imgs[1])
                else:
                    print("[data_loader.py] Unknown load_field [{}]".format(load_field))
                    exit()
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

