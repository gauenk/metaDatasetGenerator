from easydict import EasyDict as edict
from core.configBase import *
from core.configData import *


# ----------------- misc helper functions ------------------

def create_model_path(modelInfo):
    path = "output/classification/{}/".format(modelInfo.train_set,modelInfo.name)    
    return path

def create_model_name(modelInfo):
    name = "{}_{}_{}_{}_iter_{}".format(
        modelInfo.train_set,modelInfo.architecture,
        modelInfo.image_noise,modelInfo.prune,modelInfo.iterations)
    return name
    
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
                     'separability':['separability_ds_labels','separability_ds_correct']}

cfg.modelInfo = edict()
cfg.modelInfo.architecture = "lenet5"
cfg.modelInfo.iterations = 100000
cfg.modelInfo.train_set = 'cifar_10'
cfg.modelInfo.image_noise = 'yesImageNoise'
cfg.modelInfo.prune = 'noPrune'
reset_model_name()

cfg.data = edict()
cfg.data.train_imdb = "cifar_10-train-default"
cfg.data.test_imdb = "cifar_10-val-default"
cfg.data.numberOfSamples = 1000

cfg.density_estimation = edict()
cfg.density_estimation.algorithm = 'kmeans'
#cfg.density_estimation.search = [2,5,10,20,50,100]
cfg.density_estimation.search = [2,5,10]
cfg.density_estimation.nClusters = 0

# cfg.comboInfo = {"conv1-conv2":[]}
cfg.comboInfo = {'conv1','conv2','conv1-conv2'}
cfg.layerList = ['conv1','conv2']

# cfg.comboInfo = {"conv1":[]}
# cfg.layerList = ['conv1']

# variables for plotting
cfg.plot = edict()
cfg.plot.marker_list = ['o','v','^','<','>','8','s','p','h','H','+','x','X','D','d']
cfg.plot.color_dict = {'train':'b','test':'g'}


def checkConfigEquality(validConfig,proposedConfig):
    """
    check if the input config edict is the same
    as the current config edict
    """
    for key,validValue in validConfig.items(): # iterate through the "truth"
        if key not in proposedConfig.keys(): return False
        proposedValue = proposedConfig[key]
        if type(validValue) is edict or type(validValue) is dict:
            isValid = checkConfigEquality(validValue,proposedValue)
            if not isValid: return False
            continue
        if proposedValue != validValue: return False
    return True


## below is only for optimizing caching...

# ---> the parameters that change how the data is loaded originally <---

cfgForCaching = edict()
cfgForCaching.expName = cfg.expName
cfgForCaching.data = cfg.data
cfgForCaching.modelInfo = cfg.modelInfo
cfgForCaching.layerList = cfg.layerList
cfgForCaching.comboInfo = cfg.comboInfo

# cfg_clusterCache.modelInfo = cfg.modelInfo
# cfg_clusterCache.data = cfg.data
# cfg_clusterCache.train_imdb = cfg.train_imdb
# cfg_clusterCache.test_imdb = cfg.test_imdb
# cfg_clusterCache.numberOfSamples = cfg.numberOfSamples
# cfg.numberOfSamples
