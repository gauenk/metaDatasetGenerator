import numpy as np
from numpy import random as npr
from easydict import EasyDict as edict
from core.configBase import *

def createExhaustiveTranslationConfigs(translation_input_list):
    translation_list = []
    for trans in translation_input_list:
        if trans is None or trans is False:
            trans_dict = [{'step':False,'direction':False}]
        else:
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

def createExhaustiveFlipConfigs(flip_input_list):
    flip_list = []
    for flip in flip_input_list:
        flip_dict = {'flip':flip}
        flip_list.append(flip_dict)
    return flip_list

def getnumel(alist_of_lists):
    numel = 1
    if alist_of_lists is None: return numel
    for alist in alist_of_lists: numel *= len(alist)
    return numel

def createDatasetAugmentationMesh(alist_of_lists):
    return create_mesh_from_lists(alist_of_lists)

def create_mesh_from_lists(alist_of_lists,verbose=False):
    numel = getnumel(alist_of_lists)
    mesh = [ [ None for transList in alist_of_lists ] for _ in range(numel) ] 
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
                    mesh[mesh_index+repeat][transListIndex] = unique_value
    return mesh

def getDatasetAugmentationConfigurationsSubset(exhaustive_augmentation_list,n_percent,random_bool):
    total_number_of_augmentations = len(exhaustive_augmentation_list)
    number_of_augmentations = int(total_number_of_augmentations * n_percent) + 1
    if random_bool:
        npr.seed(cfg.DATASET_AUGMENTATION.RANDOM_SEED)
        indices = npr.permutation(total_number_of_augmentations)[:number_of_augmentations]
    else:
        indices = np.arange(total_number_of_augmentations)[:number_of_augmentations]
    augmentation_list = getDatasetAugmentationConfigurationsFromIndices(exhaustive_augmentation_list,indices)
    return augmentation_list,indices

def getDatasetAugmentationConfigurationsFromIndices(exhaustive_augmentation_list,indices):
    augmentation_list = [ exhaustive_augmentation_list[index] for index in indices ]
    return augmentation_list

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
    cfg.DATASET_AUGMENTATION.CONFIGS,cfg.DATASET_AUGMENTATION.CONFIG_INDICES = getDatasetAugmentationConfigurationsSubset(cfg.DATASET_AUGMENTATION.EXHAUSTIVE_CONFIGS,cfg.DATASET_AUGMENTATION.N_PERC,cfg.DATASET_AUGMENTATION.RANDOMIZE_SUBSET)
    cfg.DATASET_AUGMENTATION.SIZE = len(cfg.DATASET_AUGMENTATION.CONFIGS)
    

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
    mesh = create_mesh_from_lists([cfg.DATASET_AUGMENTATION.FLIP,
                                   cfg.DATASET_AUGMENTATION.IMAGE_TRANSLATE,
                                   cfg.DATASET_AUGMENTATION.IMAGE_ROTATE,
                                   cfg.DATASET_AUGMENTATION.IMAGE_CROP
                                   ])
    reset_dataset_augmentation_with_mesh(mesh)

def set_dataset_augmentation_config():
    if not cfg.DATASET_AUGMENTATION.BOOL:
        return
    bool_0 = (cfg.DATASET_AUGMENTATION.USE_DEFAULTS and cfg.DATASET_AUGMENTATION.SET_BY_CALLING_DATASET)
    if bool_0:
        print("[configDatasetAugmentation.py] dataset augmentation conditions are in conflict. please resolve them manually")
        print("bool_0",bool_0)
        exit()
    if cfg.DATASET_AUGMENTATION.SET_BY_CALLING_DATASET:
        set_augmentation_by_calling_dataset()
    elif cfg.DATASET_AUGMENTATION.USE_DEFAULTS:
        return
    else:
        print("[configDatasetAugmentation.py] dataset augmentation conditions are all False. please resolve them manually")
        exit()

def set_augmentation_by_calling_dataset():
    if cfg.DATASET_AUGMENTATION.USE_DEFAULTS or (cfg.DATASET_AUGMENTATION.SET_BY_CALLING_DATASET is False):
        print("the [USE_DEFAULTS] flag True. Using only default augmentation values.")
        return
    if cfg.DATASETS.CALLING_DATASET_NAME == '':
        raise ValueError("Dataset must be loaded before this function can be called")
    if cfg.DATASETS.CALLING_DATASET_NAME not in cfg.DATASET_AUGMENTATION.PRESET_ARGS_BY_SET.keys():
        print("dataset not found. using defaults.")
        return
    configs = cfg.DATASET_AUGMENTATION.PRESET_ARGS_BY_SET[cfg.DATASETS.CALLING_DATASET_NAME]
    cfg.DATASET_AUGMENTATION.IMAGE_TRANSLATE=configs['translation']
    cfg.DATASET_AUGMENTATION.IMAGE_ROTATE=configs['rotation']
    cfg.DATASET_AUGMENTATION.IMAGE_CROP=configs['crop']
    cfg.DATASET_AUGMENTATION.IMAGE_FLIP=configs['flip']
    translation_list = createExhaustiveTranslationConfigs(configs['translation'])
    rotation_list = createExhaustiveRotationConfigs(configs['rotation'])
    crop_list = createExhaustiveCropConfigs(configs['crop'])
    flip_list = createExhaustiveFlipConfigs(configs['flip'])
    mesh = create_mesh_from_lists([flip_list,translation_list,rotation_list,crop_list])
    reset_dataset_augmentation_with_mesh(mesh)

flip_input_list = [True,False]
flip_list = createExhaustiveFlipConfigs(flip_input_list)
translation_input_list = [2]
translation_input_list = [False]
translation_list = createExhaustiveTranslationConfigs(translation_input_list)
rotation_input_list = [4*i-30 for i in range(15+1)] + [0]
rotation_input_list = [False]
rotation_list = createExhaustiveRotationConfigs(rotation_input_list)
#crop_input_list = [i+1 for i in range(6)]
crop_input_list = [False]
crop_list = createExhaustiveCropConfigs(crop_input_list)
mesh = create_mesh_from_lists([flip_list,translation_list,rotation_list,crop_list])

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
        # 'translation': [2],
        # 'rotation': [5*i-45 for i in range(9+1)] + [0],
        'rotation': [-45+15*i for i in range(7)] + [-5,5],
        # 'crop': [i+1 for i in range(6)],
        'translation': [False],
        # 'rotation': [None],
        'crop': [False],
        'flip': [False]
    },
    'cifar_10': {
        # 'translation': [3],
        # 'rotation': [(2/3.)*i-5 for i in range(15+1)] + [0],
        # 'crop': [2],
        # 'flip': [False,True],
        'translation': [False],
        # 'rotation': [(2/3.)*i-5 for i in range(15+1)] + [0],
        'rotation': [0+10*i for i in range(36+1)],
        'crop': [False],
        'flip': [False]

    },
}            

# general info and set defaults
cfg.DATASET_AUGMENTATION.BOOL = True
cfg.DATASET_AUGMENTATION.N_SAMPLES = 0.25
cfg.DATASET_AUGMENTATION.RANDOMIZE = False
cfg.DATASET_AUGMENTATION.RANDOMIZE_SUBSET = False
cfg.DATASET_AUGMENTATION.RANDOM_SEED = 123
cfg.DATASET_AUGMENTATION.IMAGE_NOISE=0 #[0,1] for intensity
cfg.DATASET_AUGMENTATION.IMAGE_TRANSLATE=translation_input_list
cfg.DATASET_AUGMENTATION.IMAGE_ROTATE=rotation_input_list
cfg.DATASET_AUGMENTATION.IMAGE_CROP=crop_input_list #[0,1] for cropping; 0 = no cropping; 1 = crop to (i) bbox edge, (ii) center pixel; nominally use [0,.5]
cfg.DATASET_AUGMENTATION.IMAGE_FLIP=flip_input_list
cfg.DATASET_AUGMENTATION.N_PERC = 1.0
cfg.DATASET_AUGMENTATION.EXHAUSTIVE_CONFIGS = mesh
cfg.DATASET_AUGMENTATION.CONFIGS,cfg.DATASET_AUGMENTATION.CONFIG_INDICES = getDatasetAugmentationConfigurationsSubset(cfg.DATASET_AUGMENTATION.EXHAUSTIVE_CONFIGS,
                                                                                                                      cfg.DATASET_AUGMENTATION.N_PERC,
                                                                                                                      cfg.DATASET_AUGMENTATION.RANDOMIZE_SUBSET)
cfg.DATASET_AUGMENTATION.SIZE = len(cfg.DATASET_AUGMENTATION.CONFIGS)
cfg.DATASET_AUGMENTATION.VERSION = 'v0.1'
cfg.DATASET_AUGMENTATION.USE_DEFAULTS = False
cfg.DATASET_AUGMENTATION.SET_BY_CALLING_DATASET = True
# cfg.DATASET_AUGMENTATION.EXHAUSTIVE_CONFIGS = []
# cfg.DATASET_AUGMENTATION.CONFIGS = []
# cfg.DATASET_AUGMENTATION.SIZE = 0

