"""
This evalutates classification methods
"""

import numpy as np
from easydict import EasyDict as edict
from core.config import cfg,cfgData
from easydict import EasyDict as edict

from cls_utils import metrics_by_class,metrics_by_augmentation,print_evaluations,write_evaluation_to_csv,plot_rotation_by_class,plot_rotation_with_layers_angle
from utils.misc import computeEntropyOfNumpyArray
from cache.test_results_cache import TestResultsCache
from utils.create_angle_dataset_utils import create_angle_dataset_from_classification_eval

class classificationEvaluator(object):
    """Image database."""

    def __init__(self):
        self.imdb = None
        self.ds_loader = None
        self.output_dir = None
        self.save_cache = None
        self.agg_model_output = None


    def set_evaluation_parameters(self, imdb, ds_loader, agg_model_output, output_dir, **kwargs):
        self.imdb = imdb
        self.ds_loader = ds_loader
        self.agg_model_output = agg_model_output
        self.output_dir = output_dir
        self.save_cache = TestResultsCache(output_dir,cfg,imdb.config,None,'evaluate_classification')
        evaluation_results = None #self.save_cache.load()
        self.evaluation_results = evaluation_results
        self.other_information = {}
        if kwargs is not None:
            for key,value in kwargs.items():
                self.other_information[key] = value

    def evaluate_detections(self):
        print("# of classes: {}".format(self.imdb.num_classes))
        print("# of samples to eval: {}".format(self.ds_loader.num_samples))

        if self.evaluation_results is None:
            evaluation_results = self.format_model_outputs()
            self.save_cache.save(evaluation_results)
            self.evaluation_results = evaluation_results
        self.create_report(self.evaluation_results)

    def format_model_outputs(self):
        version = "v1"
        if version == "v1":
            return self.format_model_outputs_version1()
        elif version == "v2":
            return self.format_model_outputs_version2()

    def format_model_outputs_version1(self):
        # extract relevant variables
        imdb = self.imdb
        ds_loader = self.ds_loader
        agg_model_output = self.agg_model_output
        output_dir = self.output_dir
        model_outputs = agg_model_output.results
        guessed_classes = np.argmax(model_outputs,axis=0)
        data_augmentations = ds_loader.dataset_augmentation.configs

        # [primiary results] 1st: class, 2nd: dataset augmentation, (3rd to last): guessing class index, (2nd to last): confidence, (last): correct bool
        """
        [results ordering]:
        0.) confidence
        1.) guessed class
        2.) groundtruth class
        3.) correct_bool
        4-9.) dataset augmentation
        """
        field_names_in_order = ['confidence','guessed_class','gt_class','correct','dataset_augmentation_flip','dataset_augmentation_translate_step',
                                'dataset_augmentation_translate_direction','dataset_augmentation_rotate_angle','dataset_augmentation_crop_step']
        number_of_fields = len(field_names_in_order)
        info_by_field = np.zeros((ds_loader.num_samples,number_of_fields),dtype=np.float32)

        # main eval loop
        for _,_,sample,index in ds_loader.dataset_generator(imdb.data_loader_config,load_image=False):
            gt_class_index = sample['gt_classes'][0]
            guessed_class_index = guessed_classes[index]
            confidence = model_outputs[guessed_class_index][index]
            correct_bool = guessed_class_index == gt_class_index
            info_by_field[index,0] = confidence
            info_by_field[index,1] = guessed_class_index
            info_by_field[index,2] = gt_class_index
            info_by_field[index,3] = correct_bool
            # dataset augmentations
            if sample['aug_index'] == -1:
                info_by_field[index,4:] = 0
            else:
                augmentation = data_augmentations[sample['aug_index']]
                info_by_field[index,4] = int(augmentation[0]['flip'])
                info_by_field[index,5] = int(augmentation[1]['step'])
                info_by_field[index,6] = int(augmentation[1]['direction'])
                info_by_field[index,7] = float(augmentation[2]['angle'])
                info_by_field[index,8] = float(augmentation[3]['step'])

            print(gt_class_index,guessed_class_index,confidence)

        analysis = {'info_by_field':info_by_field,'field_names_in_order':field_names_in_order}
        return analysis
        
    def format_model_outputs_version2(self):
        # extract relevant variables
        imdb = self.imdb
        classes = imdb.classes
        ds_loader = self.ds_loader
        agg_model_output = self.agg_model_output
        output_dir = self.output_dir
        model_outputs = np.array(agg_model_output.results)
        guessed_classes = np.argmax(model_outputs,axis=0)
        data_augmentations = ds_loader.dataset_augmentation.configs

        # [primiary results] 1st: class, 2nd: dataset augmentation, (3rd to last): guessing class index, (2nd to last): confidence, (last): correct bool
        """
        [results ordering]:
        0.) ordered confidence (descending by value)
        1.) guessed class (descending by confidence value)
        2.) groundtruth class
        3.) correct_bool
        4-9.) dataset augmentation
        """
        field_names_in_order_skip_to_2 = ['gt_class','correct','dataset_augmentation_flip','dataset_augmentation_translate_step',
                                          'dataset_augmentation_translate_direction','dataset_augmentation_rotate_angle','dataset_augmentation_crop_step']
        number_of_fields = len(field_names_in_order) + 2 * len(classes)
        info_by_field = np.zeros((ds_loader.num_samples,number_of_fields),dtype=np.float32)

        # main eval loop
        print(model_outputs.shape)
        exit()
        for _,_,sample,index in ds_loader.dataset_generator(imdb.data_loader_config,load_image=False):
            gt_class_index = sample['gt_classes'][0]
            confidence_across_classes = model_outputs[:,index]
            # guessed_classes = np.argsort(confidence_across_classes)
            # confidence = confidence_across_classes[guessed_classes,index]
            guessed_class_index = guessed_classes[index]
            correct_bool = guessed_class_index == gt_class_index
            info_by_field[index,0] = confidence
            info_by_field[index,1] = guessed_class_index
            info_by_field[index,2] = gt_class_index
            info_by_field[index,3] = correct_bool
            # dataset augmentations
            if sample['aug_index'] == -1:
                info_by_field[index,4:] = 0
            else:
                augmentation = data_augmentations[sample['aug_index']]
                info_by_field[index,4] = int(augmentation[0]['flip'])
                info_by_field[index,5] = int(augmentation[1]['step'])
                info_by_field[index,6] = int(augmentation[1]['direction'])
                info_by_field[index,7] = float(augmentation[2]['angle'])
                info_by_field[index,8] = float(augmentation[3]['step'])

            print(gt_class_index,guessed_class_index,confidence)

        analysis = {'info_by_field':info_by_field,'field_names_in_order':field_names_in_order}
        return analysis

    def get_array_by_fieldname(ndarray,fieldnames,field_name):
        field_index = fieldnames.index(field_name)
        return ndarray[:,field_index]

    def create_report(self,analysis_results):
        # extract relevant variables
        imdb = self.imdb
        ds_loader = self.ds_loader
        agg_model_output = self.agg_model_output
        output_dir = self.output_dir
        info_by_field = analysis_results['info_by_field']
        field_names_in_order = analysis_results['field_names_in_order']
        augmentations = ds_loader.dataset_augmentation.configs

        # print overall accuracy
        accuracy = np.mean(info_by_field[:,3])
        entropy = computeEntropyOfNumpyArray(info_by_field[:,0])
        print("overall (accuracy: {}) (entropy: {})".format(accuracy,entropy))

        evaluation = edict()
        evaluation.data = edict()
        evaluation.data.accuracy = accuracy
        evaluation.data.entropy = entropy

        evaluation.classes = edict()
        metrics_by_class(info_by_field,evaluation.classes,imdb.classes,augmentations)

        # evaluation.augmentation = edict()
        # metrics_by_augmentation(info_by_field,evaluation.augmentation,augmentations)

        
        csv_information = edict()
        csv_information.augmentation_info = edict()
        csv_information.augmentation_info.data_order = ['accuracy','entropy']
        csv_information.augmentation_info.configs = augmentations

        augmentation_names =  ['flip','translate','rotate','crop',]
        csv_information.augmentation_info.config_order = augmentation_names
        print(csv_information.augmentation_info.config_order)
        for name,value in zip(augmentation_names,augmentations[0]):
            print(name,value)
            csv_information.augmentation_info[name] = edict()
            csv_information.augmentation_info[name].config_order = value.keys()
        print(csv_information)

        net_name = agg_model_output.save_cache.test_cache_config.modelInfo.name
        write_evaluation_to_csv(evaluation,csv_information)
        angles = sorted(cfg.DATASET_AUGMENTATION.IMAGE_ROTATE)
        plot_rotation_by_class(info_by_field,imdb.classes,angles,net_name)
        print(self.other_information['activations'].agg_obj.keys())
        if 'activations' in self.other_information.keys():
            aggActivations = self.other_information['activations']
            if 'warp_angle' in aggActivations.agg_obj.keys():
                plot_rotation_with_layers_angle(info_by_field,aggActivations,angles,imdb.classes)
        #print_evaluations(evaluation)

        if cfg.TEST.CREATE_ANGLE_DATASET:
            savename = "angle_dataset_" + imdb.imdb_str + "_" + agg_model_output.save_cache.test_cache_config.modelInfo.name
            optim_angles = create_angle_dataset_from_classification_eval(info_by_field,field_names_in_order,savename)
            print(optim_angles[:10])
        exit()


        # print("-"*50)
        # print("---> results by class <---")
        # class_field_index = field_names_in_order.index("class")
        # self.print_results_by_class(imdb,info_by_field,class_field_index)
        # print("-"*50)

        # print("-"*50)
        # print("---> results by dataset augmentation <---")
        # aug_field_index = field_names_in_order.index('dataset_augmentation')
        # self.print_results_by_augmentation(imdb,ds_loader,info_by_field,aug_field_index,class_field_index,print_by_class=False)
        # print("-"*50)

        # print("-"*50)
        # print("---> results by dataset augmentation by class <---")
        # aug_field_index = field_names_in_order.index('dataset_augmentation')
        # self.print_results_by_augmentation(imdb,ds_loader,info_by_field,aug_field_index,class_field_index,print_by_class=True)
        # print("-"*50)
        
    def print_results_by_augmentation(self,imdb,ds_loader,info_by_field,aug_field_index,class_field_index,print_by_class=False):
        for aug_index in range(-1,ds_loader.dataset_augmentation.size):
            if aug_index == -1:
                aug_info = "no augmentation"
            else:
                aug_info = ds_loader.dataset_augmentation.configs[aug_index]
            aug_indices = np.where(info_by_field[aug_field_index,:] == aug_index)[0]
            num_samples_of_aug = len(aug_indices)
            if num_samples_of_aug > 0:
                aug_acc = np.mean(info_by_field[-1,aug_indices])
            else:
                aug_acc = '-'
            if print_by_class is False:
                print("{}: acc({}) #samples({})".format(aug_info,aug_acc,num_samples_of_aug))
            else:
                print("--> [overall] {}: acc({}) #samples({}) <--".format(aug_info,aug_acc,num_samples_of_aug))
                info_by_field_by_class = info_by_field[:,aug_indices]
                self.print_results_by_class(imdb,info_by_field_by_class,class_field_index)

        
