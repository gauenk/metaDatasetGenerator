from utils.base import *
import os.path as osp
import numpy as np
from numpy import transpose as npt

def loadEvaluationRecords(classname,tp_fn_records_path,datasetCfg,saveDirPostfix=None):
    if saveDirPostfix is None:
        saveDirPostfix = "{}-{}-{}".format(datasetCfg.CALLING_DATASET_NAME,datasetCfg.CALLING_IMAGESET_NAME,datasetCfg.CALLING_CONFIG)
    saveDir = osp.join(tp_fn_records,saveDirPostfix)
    # savePath = osp.join(saveDir,"records_{}.pkl".format('det_{:s}').format(classname))
    # savePath = osp.join(saveDir,"records_{}.pkl".format('cls').format(classname))
    savePath = osp.join(saveDir,"records_{}.pkl".format(classname))
    if not osp.exists(savePath):
        print("Path [{}] doesn't exist, so there is nothing to load. quitting.".format(saveDir))
        sys.exit(1)
    print("loading evaluation records from: {}".format(savePath))
    records = readPickle(savePath)
    return records

def loadEvaluationRecordsFromPath(recordPath,load_pickle=True):
    print("loading evaluation records from: {}".format(recordPath))
    if load_pickle:
        records = readPickle(recordPath)
    else:
        records = np.load(recordPath)
    return records

def createRecordsPath(modelArchitecture,netName,imdbName,load_pickle=True):
    if load_pickle:
        recordsPath = "./output/{modelArch}/tp_fn_records/{imdbName}/records_cls_{netName}.pkl".format(modelArch=modelArchitecture,imdbName=imdbName,netName=netName)
    else:
        recordsPath = "./output/{modelArch}/tp_fn_records/{imdbName}/records_cls_{netName}.npy".format(modelArch=modelArchitecture,imdbName=imdbName,netName=netName)
    return recordsPath

def loadRecord(imdb_name,modelInfo,load_pickle=True):
    records_path = createRecordsPath(modelInfo.architecture,modelInfo.name,imdb_name,load_pickle)
    records = loadEvaluationRecordsFromPath(records_path,load_pickle)
    return records
