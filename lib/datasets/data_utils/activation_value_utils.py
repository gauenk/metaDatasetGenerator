from utils.base import *
import os.path as osp
import numpy as np

def loadActivationValues(imdb_name,layerList,netName,load_pickle=False,load_bool=False):
    dirPath = osp.join("./output/activity_vectors/classification/","{}".format(imdb_name))
    fn_prefix = "{}_{}".format(layer,netName)
    if load_bool:
        fn_bool = fn_prefix + '_bool'
        try:
            avDict = loadActivationValuesLoadLoop(dirPath,layerList,fn_bool)
        except:
            print("'boolizing' the activity vector'")
            avDict = loadActivationValuesLoadLoop(dirPath,layerList,fn_prefix)
            convertAvDictToBool(avDict)
    else:
        avDict = loadActivationValuesLoadLoop(dirPath,layerList,fn_prefix)
    return avDict

def convertAvDictToBool(avDict):
    transformations = {}
    transformations['apply_relu'] = True
    transformations['normalize'] = False
    transformations['to_bool'] = True
    return transformNumpyData(data,transformations)

def loadActivationValuesLoadLoop(dirPath,layerList,fn_prefix):
    avDict = {}
    for layer in layerList:
        if load_pickle:
            fn = osp.join(dirPath,fn_prefix+".pkl")
            avDict[layer] = readPickle(fn)
        else:
            fn = osp.join(dirPath,fn_prefix+".npy")
            avDict[layer] = np.load(fn)
    return avDict
            
def createActivationValuesComboDict(activation_values,comboInfo,numberOfSamples=-1,verbose=False):
    # init combination dictinaries
    activationCombos = {}
    for comboID in comboInfo:
        activationCombos[comboID] = []

    if numberOfSamples == -1:
        numberOfSamples = len(imageIndexList)

    # aggregate activation information via combination settings
    for comboID in comboInfo:
        layerNames = comboID.split("-")
        activationCombosList = []
        # single samples (e.g. one row; we are building the features)
        for layerName in layerNames:
            layer_activations = transformNumpyData(activation_values[layerName][:numberOfSamples,:].reshape(numberOfSamples,-1))
            activationCombosList.append(layer_activations)
        # add the single sample to the data list
        activationCombosList = np.hstack(activationCombosList)
        activationCombos[comboID] = activationCombosList

    # print info
    for comboID in comboInfo:
        if cfg.verbose or verbose:
            print("all [#samples x #ftrs]",comboID,activationCombos[comboID].shape)
        
    return activationCombos
    
