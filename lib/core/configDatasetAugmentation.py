import numpy as np
from numpy import random as npr
from easydict import EasyDict as edict
from core.configBase import *

def createExhaustiveTranslationConfigs(translation_input_list):
    translation_list = []
    for trans in translation_input_list:
        trans_dict = [{'step':trans} for _ in range(4)]
        trans_dict[0]['direction'] = 'u'
        trans_dict[1]['direction'] = 'd'
        trans_dict[2]['direction'] = 'l'
        trans_dict[3]['direction'] = 'r'
        translation_list.extend(trans_dict)
    return translation_list

def createExhaustiveCropConfigs(crop_input_list):
    crop_list = []
    for crop in crop_input_list:
        crop_dict = {'step':crop}
        crop_list.append(crop_dict)
    return crop_list

def createExhaustiveRotationConfigs(rotation_input_list):
    rotation_list = []
    for angle in rotation_input_list:
        rotation_dict = {'angle':angle}
        rotation_list.append(rotation_dict)
    return rotation_list

def getnumel(alist_of_lists):
    numel = 1
    if alist_of_lists is None: return numel
    for alist in alist_of_lists: numel *= len(alist)
    return numel

def createDatasetAugmentationMesh(alist_of_lists):
    return create_mesh_from_lists(alist_of_lists)

def create_mesh_from_lists(alist_of_lists,verbose=False):
    numel = getnumel(alist_of_lists)
    mesh = [ [ None for _ in range(numel) ] for transList in alist_of_lists ]
    for transListIndex,transList in enumerate(alist_of_lists):
        assert (numel % len(transList)) == 0,"translation index zero"
        numberOfRepeats = numel // len(transList)
        numberOfUniqueValues = len(transList)
        alist_of_lists_copy = alist_of_lists[:]
        for pop_index in range(transListIndex+1): alist_of_lists_copy.pop(0)
        numberTogether = getnumel(alist_of_lists_copy)
        numberOfBlocks = numberOfRepeats // numberTogether
        blockSpacing = numberTogether * numberOfUniqueValues
        if verbose:
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-")
            print("transListIndex: {}".format(transListIndex))
            print("numberOfRepeats: {}".format(numberOfRepeats))
            print("numberOfUniqueValues: {}".format(numberOfUniqueValues))
            print("numberTogether: {}".format(numberTogether))
            print("numberOfBlocks: {}".format(numberOfBlocks))
        for start_index,unique_value in enumerate(transList):                
            for block_index in range(numberOfBlocks):
                mesh_index = start_index * numberTogether + block_index * blockSpacing
                for repeat in range(numberTogether):
                    mesh[transListIndex][mesh_index+repeat] = unique_value
    return mesh

def getDatasetAugmentationConfigurationsRandomSubset(da_config,n_percent):
    nconfigs = int(len(da_config[0]) * n_percent) + 1
    if cfg.DATASET_AUGMENTATION.RANDOMIZE: indicies = npr.permutation(len(da_config[0]))[:nconfigs]
    else: indicies = np.arange(len(da_config[0]))[:nconfigs]
    da_cfg = [ [ transList[index] for index in indicies] for transList in da_config ]
    return da_cfg

def getDatasetAugmentationConfigurationsFromIndices(da_config,da_inds):
    da_cfg = [ [ transList[index] for index in da_inds] for transList in da_config ]
    return da_cfg

def formatDatasetAugmentationForGetRawCroppedImage(input_da_config,da_inds):
    if type(da_inds) is not list: da_inds = [da_inds]
    da_config = getDatasetAugmentationConfigurationsFromIndices(input_da_config,da_inds)
    n_transform_types = len(da_config)
    n_transforms = len(da_config[0])
    output = [{'transformations':[None for _ in range(n_transform_types)]} \
              for i in range(n_transforms) ]
    for i in range(n_transforms):
        for j in range(n_transform_types):
            output[i]['transformations'][j] = da_config[j][i]
    # print(output)
    return output

def reset_dataset_augmentation_with_mesh(mesh):
    cfg.DATASET_AUGMENTATION.EXHAUSTIVE_CONFIGS = mesh
    cfg.DATASET_AUGMENTATION.CONFIGS = getDatasetAugmentationConfigurationsRandomSubset(cfg.DATASET_AUGMENTATION.EXHAUSTIVE_CONFIGS,cfg.DATASET_AUGMENTATION.N_PERC)
    cfg.DATASET_AUGMENTATION.SIZE = len(cfg.DATASET_AUGMENTATION.CONFIGS[0])
    

def reset_dataset_augmentation(new_list,transformation_type):
    if transformation_type == 'translation':
        translation_list = createExhaustiveTranslationConfigs(translation_input_list)
        cfg.DATASET_AUGMENTATION.IMAGE_TRANSLATE = translation_list
    elif transformation_type == 'rotation':
        rotation_list = createExhaustiveRotationConfigs(rotation_input_list)
        cfg.DATASET_AUGMENTATION.IMAGE_ROTATE = rotation_list
    elif transformation_type == 'crop':
        crop_list = createExhaustiveCropConfigs(crop_input_list)
        cfg.DATASET_AUGMENTATION.IMAGE_CROP = crop_list
    mesh = create_mesh_from_lists([cfg.DATASET_AUGMENTATION.IMAGE_TRANSLATE,
                                   cfg.DATASET_AUGMENTATION.IMAGE_ROTATE,
                                   cfg.DATASET_AUGMENTATION.IMAGE_CROP])
    reset_dataset_augmentation_with_mesh(mesh)

def set_augmentation_by_calling_dataset():
    if cfg.DATASETS.CALLING_DATASET_NAME == '':
        raise ValueError("Dataset must be loaded before this function can be called")
    configs = cfg.DATASET_AUGMENTATION.PRESET_ARGS_BY_SET[cfg.DATASETS.CALLING_DATASET_NAME]
    translation_list = createExhaustiveTranslationConfigs(configs['translation'])
    rotation_list = createExhaustiveRotationConfigs(configs['rotation'])
    crop_list = createExhaustiveCropConfigs(configs['crop'])
    mesh = create_mesh_from_lists([rotation_list,translation_list,crop_list])
    reset_dataset_augmentation_with_mesh(mesh)

translation_input_list = [2]
translation_list = createExhaustiveTranslationConfigs(translation_input_list)
rotation_input_list = [4*i-30 for i in range(15+1)] + [0]
rotation_list = createExhaustiveRotationConfigs(rotation_input_list)
crop_input_list = [i+1 for i in range(6)]
crop_list = createExhaustiveCropConfigs(crop_input_list)
mesh = create_mesh_from_lists([rotation_list,translation_list,crop_list])

# dataset augmentation
cfg.DATASET_AUGMENTATION = edict()

# 86% (00% train | 100% test) @ 6000 iters @ mini-batch size 512
# 88% (10% train | 100% test) @ 9000 iters @ mini-batch size 512
# 

# only train on boundary of the space of positive samples to get perfect model

# static choises for some datasets
cfg.DATASET_AUGMENTATION.PRESET_ARGS_BY_SET = {
    # 'mnist': {
    #     'translation': [0],
    #     'rotation': [i for i in range(1)],
    #     'crop': [i for i in range(1)]
    # },
    'mnist': {
        'translation': [2],
        'rotation': [4*i-30 for i in range(15+1)] + [0],
        'crop': [i+1 for i in range(6)]
    },
    'cifar_10': {
        'translation': [3],
        'rotation': [(2/3)*i-5 for i in range(15+1)] + [0],
        'crop': [2]
    },
}            

# general info and set defaults
cfg.DATASET_AUGMENTATION.BOOL = True
cfg.DATASET_AUGMENTATION.N_SAMPLES = 0.25
cfg.DATASET_AUGMENTATION.RANDOMIZE = False
cfg.DATASET_AUGMENTATION.IMAGE_NOISE=0 #[0,1] for intensity
cfg.DATASET_AUGMENTATION.IMAGE_TRANSLATE=translation_input_list
cfg.DATASET_AUGMENTATION.IMAGE_ROTATE=rotation_input_list
cfg.DATASET_AUGMENTATION.IMAGE_CROP=crop_input_list #[0,1] for cropping; 0 = no cropping; 1 = crop to (i) bbox edge, (ii) center pixel; nominally use [0,.5]
cfg.DATASET_AUGMENTATION.N_PERC = 1.0
cfg.DATASET_AUGMENTATION.EXHAUSTIVE_CONFIGS = mesh
cfg.DATASET_AUGMENTATION.CONFIGS = getDatasetAugmentationConfigurationsRandomSubset(cfg.DATASET_AUGMENTATION.EXHAUSTIVE_CONFIGS,cfg.DATASET_AUGMENTATION.N_PERC)
cfg.DATASET_AUGMENTATION.SIZE = len(cfg.DATASET_AUGMENTATION.CONFIGS[0])
# cfg.DATASET_AUGMENTATION.EXHAUSTIVE_CONFIGS = []
# cfg.DATASET_AUGMENTATION.CONFIGS = []
# cfg.DATASET_AUGMENTATION.SIZE = 0

