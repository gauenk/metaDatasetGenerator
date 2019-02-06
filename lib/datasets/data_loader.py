import cv2,copy
import numpy as np
import numpy.random as npr
from easydict import EasyDict as edict
from utils.base import readNdarray,print_warning
from utils.blob import preprocess_image_for_model,im_list_to_blob,siameseImagesToBlobs
from utils.image_utils import save_image_list_to_file
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

        self.transform_each_sample.data_list = []
        for data_info_from_init in transform_each_sample.DATA_LIST:
            data_info = edict()
            print(data_info_from_init)
            data_info.bool = data_info_from_init.BOOL
            data_info.rand = data_info_from_init.RAND
            data_info.type = data_info_from_init.TYPE
            data_info.params = data_info_from_init.PARAMS
            self.set_transform_data_info(data_info) 
            self.transform_each_sample.data_list.append(data_info)

        self.transform_each_sample.label_list = []
        for label_info_from_init in transform_each_sample.LABEL_LIST:
            label_info = edict()
            label_info.bool = label_info_from_init.BOOL
            label_info.type = label_info_from_init.TYPE
            label_info.params = label_info_from_init.PARAMS
            self.set_transform_label_params_by_type(label_info)
            self.transform_each_sample.label_list.append(label_info)

    def set_transform_label_params_by_type(self,label_info):
        if label_info.bool and label_info.type == "file_replace":
            filename = label_info.params['filename']
            new_labels = readNdarray(filename)
            label_info.params['labels'] = new_labels
            assert self.num_samples == len(new_labels), "the replacing labels size needs to be equal"
        
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

    def set_transform_data_info(self,data_info):
        if data_info.bool is False:
            return
        if data_info.rand:
            transform_cache_postfix = "{}_index".format(data_info.type)
            roidb_cache_fieldname = "{}_{}".format(self.roidb_cache_fieldname_prefix,transform_cache_postfix)
            data_info.params['roidb_info'] = self.roidb_cache.load(roidb_cache_fieldname)
            if data_info.params['roidb_info'] is None:
                print_warning(__file__,"new roidb_info for transform data")
                roidb_info = self.get_transform_data_roidb_info_by_type(data_info.type,data_info.params)
                data_info.params['roidb_info'] = roidb_info
                self.roidb_cache.save(roidb_cache_fieldname,roidb_info)
        else:
            print("[data_loader] transform each sample is not random... lame. Quitting.")
            exit()
        
    def get_transform_data_roidb_info_by_type(self,transform_type,params):
        # what does this code even do? Why is "rotate_index" an integer between 0 and 360?
        if transform_type == 'rotate':
            angle_min = params['angle_min']
            angle_max = params['angle_max']
            rotate_angle = (npr.rand(self.roidb_size) * (angle_max - angle_min)) + angle_min
            return rotate_angle
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
        
    def add_data_transform_information(self,sample,roidb_index):
        # add info for transforming data
        sample['data_transform_info'] = []
        sample['data_replace_label'] = False
        for data_info in self.transform_each_sample.data_list:
            if data_info.bool is False:
                continue
            sample['data_transform_info'].append(data_info.params['roidb_info'][roidb_index])

    def apply_sample_label_transform(self,sample,roidb_index,index):
        for label_info in self.transform_each_sample.label_list:
            #print(label_info)
            if label_info.bool is False:
                continue
            if label_info.type == "file_replace":
                new_labels = label_info.params['labels']
                index_type = label_info.params['index_type']
                index_dict = {'roidb_index':roidb_index,'index':index}
                sample['gt_classes'] = new_labels[index_dict[index_type]]
            elif label_info.type == "angle":
                angle_index = label_info.params['angle_index']
                transforms = self.dataset_augmentation.configs[sample['aug_index']]
                sample['gt_classes'] = transforms[angle_index]['angle'] / 360.
            elif label_info.type == "normalize":
                min_val,max_val = label_info.params['min'],label_info.params['max']
                data_min_val,data_max_val = label_info.params['data_min'],label_info.params['data_max']
                # 1.) shift interval from [a,b] to [0,b-a]
                gt_labels = sample['gt_classes'] - data_min_val
                # 2.) shift interval from [0,b-a] to [0,1]
                gt_labels = gt_labels/(data_max_val - data_min_val)
                # 3.) change [0,1] to [min_val,max_val]
                sample['gt_classes'] = (gt_labels * (max_val - min_val)) + min_val
            elif label_info.type == "data_replace":
                load_settings = label_info.params.load_settings
                load_as_blob = label_info.params.load_as_blob
                load_image = label_info.params.load_image
                label_image,_ = self.load_sample(sample,load_settings,load_as_blob=load_as_blob,load_image=load_image)
                # save_image_list_to_file([label_image],None,size=32,infix="dataloader_")
                sample['gt_classes'] = label_image
            else:
                print("[data_loader] unknown transformation label.type of each sample [{}]. quitting.".format(label_info.type))
                exit()

    def apply_sample_data_transformation(self,sample,img):
        if img is None:
            return 
        trans_img = img
        for info_index,data_info in enumerate(self.transform_each_sample.data_list):
            if data_info.bool is False:
                continue
            if data_info.type == 'rotate':
                angle = sample['data_transform_info'][info_index]
                trans_img,_ = rotateImage(trans_img,angle)
            else:
                print("[data_loader] unknown transformation data.type of each sample [{}]. quitting.".format(data_info.type))
                exit()
        return trans_img

    def get_sample_with_info(self,index):
        sample,roidb_index = self.add_dataset_augmentation_information(index)
        self.add_records_information(sample,roidb_index)
        self.add_image_id_information(sample,roidb_index)
        self.add_data_transform_information(sample,roidb_index)
        self.apply_sample_label_transform(sample,roidb_index,index)
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

# /home/gauenk/Documents/experiments/metaDatasetGenerator/output/classification/cifar_10/cifar_10_train_default_lenet5_sgd_noImageNoise_noPrune_noDsAug_yesClassFilter2_iter_10500/lookup_agg_model_output.pkl

# /home/gauenk/Documents/experiments/metaDatasetGenerator/output/classification/cifar_10/cifar_10_train_default_lenet5_sgd_noImageNoise_noPrune_noDsAug_yesClassFilter2_iter_10500/a0a52333-5888-46af-86a0-b9b3d70246ec.pkl

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


    #
    # primary loading functions
    #

    def load_sample(self,sample,load_settings,load_as_blob=False,load_image=True):
        if load_image is False:
            return None,None
        label = sample['gt_classes']
        image_path = self.imdb.image_path_at(sample['image_id'])
        img = cv2.imread(image_path)
        image_shape = img.shape
        if load_settings.color_bool == False:
           img = img[:,:,0][:,:,np.newaxis]
        scales = []
        if load_settings.cropped_to_box_bool:
            img = cropImageToAnnoRegion(img,sample['boxes'][load_settings.cropped_to_box_index]) 
        self.apply_sample_data_transformation(sample,img)
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

        if load_settings.color_bool == False:
            img_shape = img.shape + (-1,)
            img = img.reshape(img_shape)
        return self.format_return_values(img,label,scales,load_settings)

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

