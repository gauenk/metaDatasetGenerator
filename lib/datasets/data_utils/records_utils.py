from utils.base import *
import os.path as osp
import numpy as np
from numpy import transpose as npt

def loadRecordsAndReorderToImageIndex(imdb_name,modelInfo,imageIndexIDs,load_record_pickle=False):
    loaded_records = loadRecord(imdb_name,modelInfo,cfg.load_record_pickle)
    if load_record_pickle:
        records = np.zeros((len(imageIndexIDs)),dtype=np.uint8)
        for record_index,imageIndex in enumerate(imageIndexIDs):
            records[record_index] = loaded_records[imageIndex][0]
    else:
        records = loaded_records.astype(np.uint8)
    return records

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

def loadEvaluationRecordsFromPath(recordPath,load_pickle=False):
    print("loading evaluation records from: {}".format(recordPath))
    if load_pickle:
        records = readPickle(recordPath)
    else:
        records = np.load(recordPath)
    return records

def createRecordsPath(modelArchitecture,netName,imdbName,load_pickle=False):
    if load_pickle:
        recordsPath = "./output/{modelArch}/tp_fn_records/{imdbName}/records_cls_{netName}.pkl".format(modelArch=modelArchitecture,imdbName=imdbName,netName=netName)
    else:
        recordsPath = "./output/{modelArch}/tp_fn_records/{imdbName}/records_cls_{netName}.npy".format(modelArch=modelArchitecture,imdbName=imdbName,netName=netName)
    return recordsPath

def loadRecord(imdb_name,modelInfo,load_pickle=False):
    records_path = createRecordsPath(modelInfo.architecture,modelInfo.name,imdb_name,load_pickle)
    records = loadEvaluationRecordsFromPath(records_path,load_pickle)
    return records
