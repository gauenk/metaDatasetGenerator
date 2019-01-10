from easydict import EasyDict as edict
from core.configBase import *
from core.configData import *
from copy import deepcopy


# ----------------- misc helper functions ------------------

def create_model_path(modelInfo):
    path = "output/classification/{}/".format(modelInfo.train_set,modelInfo.name)    
    return path

def create_model_name_type_c(modelInfo):
    name = "{}_{}_{}_{}_{}_{}_iter_{}".format(
        modelInfo.train_set,modelInfo.architecture,modelInfo.optim,
        modelInfo.image_noise,modelInfo.prune,
        modelInfo.dataset_augmentation,modelInfo.iterations)
    return name

def create_model_name_type_a(modelInfo):
    name = "{}_{}_{}_{}_{}_aug25perc_iter_{}".format(
        modelInfo.train_set,modelInfo.architecture,modelInfo.optim,
        modelInfo.image_noise,modelInfo.prune,modelInfo.iterations)
    return name

def create_model_name_type_b(modelInfo):
    name = "{}_{}_{}_{}_aug25perc_iter_{}".format(
        modelInfo.train_set,modelInfo.architecture,
        modelInfo.image_noise,modelInfo.prune,modelInfo.iterations)
    return name

def create_model_name(modelInfo):
    name = create_snapshot_prefix(modelInfo)
    name += '_iter_' + str(modelInfo.iterations)
    return name
    # if 'optim' in modelInfo.keys() and 'dataset_augmentation' in modelInfo.keys():
    #     return create_model_name_type_c(modelInfo)
    # elif 'optim' in modelInfo.keys():
    #     return create_model_name_type_a(modelInfo)
    # else:
    #     return create_model_name_type_b(modelInfo)        
    
def reset_model_name():
    cfg.modelInfo.name = create_model_name(cfg.modelInfo)
    cfg.modelInfo.path = create_model_path(cfg.modelInfo)


## ------------------- START HERE -----------------------

__C = edict()
cfg = __C

cfg.expName = 'default'
cfg.verbose = True
cfg.cache = True
cfg.measure_name_dict = {'cluster':['silhouette','homogeneity_ds_labels','homogeneity_correct'],
                     'separability':['separability_ds_labels','separability_ds_correct','separability_ds_label_final_layer_only']}
cfg.special_plot_name_dict = {'separability_ds_label_final_layer_only':['train_with_gen_error','test_with_gen_error']}
cfg.load_av_pickle = False
cfg.load_record_pickle = False
cfg.skip_silhouette_score = True

# actually for the experiment
cfg.load_bool_activity_vectors = False

cfg.modelInfo = edict()
cfg.modelInfo.architecture = "lenet5"
cfg.modelInfo.iterations = 40000 #100000
cfg.modelInfo.train_set = 'cifar_10'
cfg.modelInfo.image_noise = 'yesImageNoise'
cfg.modelInfo.prune = 'noPrune'
cfg.modelInfo.optim = 'sgd'
cfg.modelInfo.dataset_augmentation = 'noDsAug'
cfg.modelInfo.classFilter = False
reset_model_name()

cfg.data = edict()
cfg.data.train_imdb = "cifar_10-train-default"
cfg.data.test_imdb = "cifar_10-val-default"
cfg.data.numberOfSamples = -1
cfg.data.transformations = edict()
cfg.data.transformations.apply_relu = False
cfg.data.transformations.normalize = False
cfg.data.transformations.to_bool = False
assert not( (cfg.data.transformations.normalize is True) and (cfg.data.transformations.to_bool is True)), "we can't have both normalize and bool transforms"

cfg.density_estimation = edict()
cfg.density_estimation.algorithm = 'kmeans'
# cfg.density_estimation.search = [2,10,50,100,200,500,1000] (5k_results suggests ~200)
# cfg.density_estimation.search = [2,5,10,35,50,100,200,350,500]
# cfg.density_estimation.search = [2,5,10,35,50,100,200]
# cfg.density_estimation.search = [2,5,10,20,30,40,50,60,70,80,90,100]
cfg.density_estimation.search = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]

cfg.density_estimation.nClusters = cfg.density_estimation.search[-1]

# cfg.comboInfo = ['conv1','conv2','ip1','cls_score','cls_prob',
#                  'conv1-conv2','conv1-ip1','conv1-cls_score','conv1-cls_prob',
#                  'conv2-ip1','conv2-cls_score','conv2-cls_prob',
#                  'ip1-cls_score','ip1-cls_prob','cls_score-cls_prob']
# cfg.layerList = ['conv1','conv2','ip1','cls_score','cls_prob']

cfg.comboInfo = ['conv2','ip1','cls_score','cls_prob',
                 'conv2-ip1','conv2-cls_score','conv2-cls_prob',
                 'ip1-cls_score','ip1-cls_prob','cls_score-cls_prob']
cfg.layerList = ['conv2','ip1','cls_score','cls_prob']

# cfg.comboInfo = ['conv1','conv1-cls_prob']
# cfg.layerList = ['conv1','cls_prob']


# variables for plotting
cfg.plot = edict()
cfg.plot.marker_list = ['o','v','^','<','>','8','s','p','h','H','+','x','X','D','d']
cfg.plot.color_dict = {'train':'b','test':'g'}


def update_config(input_cfg,experiment_config):
    for key,value in experiment_config.items():
        if key not in input_cfg.keys(): raise ValueError("key [{}] not in original configuration".format(key))
        if type(value) is edict: update_config(input_cfg[key],value)
        else: input_cfg[key] = value
    reset_caching_config()

def find_first_config_difference(new_cfg,old_cfg):
    for key,old_value in old_cfg.items():
        if key is 'name': continue
        new_value = new_cfg[key]
        if key not in new_cfg.keys(): raise ValueError("key [{}] not in original configuration".format(key))
        if type(old_value) is edict: 
            a,b = find_first_config_difference(new_value,old_value)
            if a is not None: return a,b
        if old_value != new_value: return key,new_value
    return None,None

def update_one_current_config_field(exp_config):
    old_cfg = deepcopy(cfg)
    update_config(cfg,exp_config)
    fieldname,fieldvalue = find_first_config_difference(cfg,old_cfg)
    return fieldname,fieldvalue

def reset_caching_config():
    cfgForCaching.expName = cfg.expName
    cfgForCaching.data = cfg.data
    cfgForCaching.modelInfo = cfg.modelInfo
    cfgForCaching.layerList = cfg.layerList
    cfgForCaching.comboInfo = cfg.comboInfo
    cfgForCaching.density_estimation = cfg.density_estimation

def get_config_field_update_template(init_val,*args):
    template_root = edict()
    template = template_root
    for index,field in enumerate(args):
        if (index+1) == len(args): template[field] = init_val
        else:
            template[field] = edict()
            template = template[field]
    return template_root

## below is only for optimizing caching...

# ---> the parameters that change how the data is loaded originally <---

cfgForCaching = edict()
cfgForCaching.expName = cfg.expName
cfgForCaching.data = cfg.data
cfgForCaching.modelInfo = cfg.modelInfo
cfgForCaching.layerList = cfg.layerList
cfgForCaching.comboInfo = cfg.comboInfo
cfgForCaching.density_estimation = cfg.density_estimation

# cfg_clusterCache.modelInfo = cfg.modelInfo
# cfg_clusterCache.data = cfg.data
# cfg_clusterCache.train_imdb = cfg.train_imdb
# cfg_clusterCache.test_imdb = cfg.test_imdb
# cfg_clusterCache.numberOfSamples = cfg.numberOfSamples
# cfg.numberOfSamples
