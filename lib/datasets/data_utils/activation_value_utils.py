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
            
