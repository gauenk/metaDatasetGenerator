#!/usr/bin/env python2
import os,sys,re,yaml,subprocess,pickle,argparse,collections,uuid
from pprint import pprint as pp
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets.ds_utils import loadEvaluationRecordsFromPath
from core.train import get_training_roidb
from core.config import cfg as cfgOg
from core.config import cfg_from_file as og_cfg_from_file
from core.routingConfig import cfg, cfg_from_file, initRouteConfig, imdbFromDatasetDict, createRecordsPath, getResultsBaseFilenameRouting, packClassificationInformation, createClassificationExperimentCacheName, checkConfigEquality
#from core.config import cfgRouting as cfgRouting
from sklearn import svm
from sklearn import cluster as sk_cluster
from utils.base import readPickle,writePickle
from utils.misc import get_roidb
from utils.cluster import *
from utils.routing import *

# LAYERS = ["conv1","ip1","cls_score"]
# LAYERS = ["conv1","conv2","ip1","cls_score"]
#LAYERS = ["ip1","cls_score"]
# LAYERS = ["conv1","cls_score"]
# LAYERS = ["conv1"]

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Routing analysis tool')
    parser.add_argument('--cfgBase', dest='cfgBase_file',
                        help='the OG config file', default=None, type=str)
    parser.add_argument('--cfgRoute', dest='cfgRoute_file',
                        help='the routing config file', default=None, type=str)
    parser.add_argument('--cfgDensity', dest='cfgDensity_file',
                        help='the config file for density estimation',
                        default="./experiments/cfgs/routing/cifar_10-byClass-densityEstimation.yml",type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def loadActivityVectors(datasetName,imageSet,layerList,netName):
    dirPath = osp.join("./output/activity_vectors/classification/",
                   "{}-{}".format(datasetName,imageSet))
    avDict = {}
    for layer in layerList:
        fn = osp.join(dirPath,"{}_{}".format(layer,netName)+".pkl")
        avDict[layer] = readPickle(fn)
    return avDict

def splitDataIntoTrainTest(dataOg,labelsOg,dataClsOg,trainSizePerc):
    testSizePerc = 1 - trainSizePerc

    originalSize = len(dataOg)
    trainSize = int(originalSize * trainSizePerc)
    testSize = int(originalSize * testSizePerc)
    if trainSize + testSize > originalSize:
        trainSize -= 1
    elif trainSize + testSize < originalSize:
        trainSize += 1
    dataTrain = dataOg[:trainSize]
    dataTest = dataOg[testSize:]
        
    labelsTrain = labelsOg[:trainSize]
    labelsTest = labelsOg[testSize:]

    dataClsTrain = dataClsOg[:trainSize]
    dataClsTest = dataClsOg[testSize:]

    return dataTrain,labelsTrain,dataClsTrain,dataTest,labelsTest,dataClsTest

def loadDatasets(trainInfo,testInfo,clsIndex):
    dataTrain,labelsTrain,dataClsTrain = activationValuesToSVMFormat(trainInfo)
    dataTest,labelsTest,dataClsTest = activationValuesToSVMFormat(testInfo)
    # useTrainSplitAsTest = False
    # if useTrainSplitAsTest == 0:

    # else:
    #     dataTrain,labelsTrain,dataClsTrain,\
    #     dataTest,labelsTest,dataClsTest = \
    #     splitDataIntoTrainTest(dataTrain,labelsTrain,dataClsTrain,useTrainSplitAsTest)
    print("[loadDatasets] {{before clsIndex}} len(dataTrain):{} | len(dataTest): {}"\
          .format(len(dataTrain),len(dataTest)))
    if clsIndex is not None:
        selIndex = np.where(dataClsTrain == clsIndex)[0]
        dataTrain = dataTrain[selIndex,:]
        labelsTrain = labelsTrain[selIndex]

        selIndex = np.where(dataClsTest == clsIndex)[0]
        dataTest = dataTest[selIndex,:]
        labelsTest = labelsTest[selIndex]
    print("[loadDatasets] {{after clsIndex}} len(dataTrain):{} | len(dataTest): {}"\
          .format(len(dataTrain),len(dataTest)))
    return dataTrain,labelsTrain,dataClsTrain,dataTest,labelsTest,dataClsTest

def classificationExperiment(trainInfo,testInfo,clsName):
    clsName = trainInfo['clsName']
    clsIndex = None
    if clsName is not None:
        clsIndex = trainInfo['imdb'].classes.index(clsName)
        print("[SVM] Route data w.r.t class name: {}".format(clsName))

    # run experiment or load from cache
    loadExperiment = loadClsExpCache()
    if loadExperiment:
        dataTrain,labelsTrain,dataClsTrain,dataTest,labelsTest,dataClsTest,clf = loadExperiment
    else:
        dataTrain,labelsTrain,dataClsTrain,dataTest,labelsTest,dataClsTest,clf = newClassificationExperiment(trainInfo,testInfo,clsIndex)
    printDatasetInformation(dataTrain,labelsTrain,dataTest,labelsTest)

    #layerSizes = [cfg.clusterParams['nClusters']+1]
    #plotDataWithLabels(dataTrain,labelsTrain,LAYERS,layerSizes,vis=True)

    # testing data fit
    predict = clf.predict(dataTest)
    probability = False
    reportClassificationExperimentResults(predict,labelsTest,clsName,dataClsTest,probability,"test","test",testInfo['imdb'])
    
    # training data fit
    predict = clf.predict(dataTrain)
    reportClassificationExperimentResults(predict,labelsTrain,clsName,dataClsTrain,probability,"train","train",trainInfo['imdb'])


def newClassificationExperiment(trainInfo,testInfo,clsIndex):
    print("running [newClassificationExperiment]")

    print("format data for model")
    # format data for svm model
    dataTrain,labelsTrain,dataClsTrain,dataTest,labelsTest,dataClsTest = loadDatasets(trainInfo,testInfo,clsIndex)

    # train svm model
    probability = False
    clf = svm.SVC(gamma='auto',probability=probability,kernel='linear',class_weight='balanced')
    clf.fit(dataTrain,labelsTrain)

    # save cache
    dataToSave = [dataTrain,labelsTrain,dataClsTrain,dataTest,labelsTest,dataClsTest,clf]
    saveClsExpCache(dataToSave)

    return dataTrain,labelsTrain,dataClsTrain,dataTest,labelsTest,dataClsTest,clf
    
def printDatasetInformation(dataTrain,labelsTrain,dataTest,labelsTest): 
    print("train data information")
    print(dataTrain.shape,labelsTrain.shape)
    pos = np.sum(labelsTrain == 1)
    neg = np.sum(labelsTrain == 0)
    print("# pos: {}".format(pos))
    print("# neg: {}".format(neg))
    print("test data information")
    print(dataTest.shape,labelsTest.shape)
    pos = np.sum(labelsTest == 1)
    neg = np.sum(labelsTest == 0)
    print("# pos: {}".format(pos))
    print("# neg: {}".format(neg))

def writeDatasetInformation(fid,dataTrain,labelsTrain,dataTest,labelsTest): 
    myStr = ''
    myStr += "train data information"
    myStr += "{},{}".format(dataTrain.shape,labelsTrain.shape)
    pos = np.sum(labelsTrain == 1)
    neg = np.sum(labelsTrain == 0)
    myStr += "# pos: {}".format(pos)
    myStr += "# neg: {}".format(neg)
    myStr += "test data information"
    myStr += "{},{}".format(dataTest.shape,labelsTest.shape)
    pos = np.sum(labelsTest == 1)
    neg = np.sum(labelsTest == 0)
    myStr += "# pos: {}".format(pos)
    myStr += "# neg: {}".format(neg)
    fid.write(myStr)

def selectClsReferenceRoute(routeAll,routePos,routeNeg,selStr):
    if selStr == 'All': return routeAll
    elif selStr == 'Pos':return routePos
    elif selStr == 'Neg':return routeNeg
    else:
        print("route type {} not found.".format(selStr))
        return None
    
def saveClsExpCache(saveData):
    path = createClassificationExperimentCacheName()
    cache = readPickle(path)
    if cache is None: cache = {}
    uuID = str(uuid.uuid4())
    blob = {'config':cfg}
    blob['data'] = saveData
    cache[uuID] = blob
    writePickle(path,cache)

def loadClsExpCache():
    path = createClassificationExperimentCacheName()
    cache = readPickle(path)
    if cache is None or len(cache) == 0: return None
    for uuID,expData in cache.items():
        isEqual = checkConfigEquality(expData['config'],cfg)
        if isEqual: return expData['data']

def viewClsExpCache():
    path = createClassificationExperimentCacheName()
    cache = readPickle(path)
    print(cache.keys())
    for key,value in cache.items():
        print(key,value['config'].expName,checkConfigEquality(value['config'],cfg))

def prepareClassificationExperimentInfo(avA,recordsA,imdbA,avB,recordsB,imdbB,refRoute):
    clsExpClassName = cfg.loadOnlyClsStr
    layerOrder = sorted(avA.keys())
    trainSize = cfg.routingAnalysisInfo.train.size
    testSize = cfg.routingAnalysisInfo.test.size
    trainInfo = packClassificationInformation(avA,recordsA,layerOrder,refRoute,trainSize,clsExpClassName,imdbA)
    testInfo = packClassificationInformation(avB,recordsB,layerOrder,refRoute,testSize,clsExpClassName,imdbB)
    return trainInfo,testInfo

if __name__ == "__main__":
    
    # init the config information
    args = parse_args()
    cfgBaseFn,cfgRouteFn,cfgDensityFn = args.cfgBase_file,args.cfgRoute_file,args.cfgDensity_file
    print(cfgBaseFn,cfgRouteFn,cfgDensityFn)
    og_cfg_from_file(cfgBaseFn)
    cfg_from_file(cfgRouteFn)
    initRouteConfig(cfgDensityFn)

    # print(cfg.routingAnalysisInfo.comboInfo)
    # sys.exit()
    # init local params
    comboInfo = cfg.routingAnalysisInfo.comboInfo
    netName = cfg.netInfo.modelName
    modelType = cfg.netInfo.modelType
    imdbNameA = imdbFromDatasetDict(cfg.routingAnalysisInfo.train)
    imdbNameB = imdbFromDatasetDict(cfg.routingAnalysisInfo.test)
    recordsPathA = createRecordsPath(modelType,imdbNameA,netName)
    recordsPathB = createRecordsPath(modelType,imdbNameB,netName)
    layerList = cfg.routingAnalysisInfo.layers
    print("base filename: {}".format(getResultsBaseFilenameRouting()))
    print("cluster filename: {}".format(getClusterCacheFilename("clusterCacheStr")))


    # view cache for the classification experiments
    # viewClsExpCache()
    # sys.exit()


    # load the imdbs
    imdbA,roidbA = get_roidb(imdbNameA)
    imdbB,roidbB = get_roidb(imdbNameB)

    # load the records
    recordsA = loadEvaluationRecordsFromPath(recordsPathA)
    recordsB = loadEvaluationRecordsFromPath(recordsPathB)

    # plot # of 1's and 0's per class per record distribution
    plotRecordDistribution = False
    if plotRecordDistribution:
        recordDistributionPerClassPerSampleSize(recordsA,imdbA)
        recordDistributionPerClassPerSampleSize(recordsB,imdbB)

    # load the activity vectors from the original model
    dsNameA,dsSplitA = imdbNameA.split('-')[0],imdbNameA.split('-')[1]
    dsNameB,dsSplitB = imdbNameB.split('-')[0],imdbNameB.split('-')[1]
    avA = loadActivityVectors(dsNameA,dsSplitA,layerList,netName)
    avB = loadActivityVectors(dsNameB,dsSplitB,layerList,netName)
    
    # create the routes for the 'training' dataset (equivalent to "A")
    routeA,routeAPos,routeANeg = routeActivityVectorsByClass(avA,imdbA,roidbA,records=recordsA,numSamples=-1)

    # prepare information for classification experiment
    refRoute = selectClsReferenceRoute(routeA,routeAPos,routeANeg,cfg.clsExperimentInfo.referenceRoute.referenceName)
    trainInfo,testInfo = prepareClassificationExperimentInfo(avA,recordsA,imdbA,avB,recordsB,imdbB,refRoute)
    clsExpClassName = cfg.loadOnlyClsStr
    classificationExperiment(trainInfo,testInfo,clsExpClassName)

    #fitSvmModel(avA,avB,recordsA,recordsB,routeA)
    sys.exit()


    routeB,routeBPos,routeBNeg = routeActivityVectorsByClass(avB,imdbB,roidbB,records=recordsB)

    indexWeightStr = None
    #indexWeightStr = 'descending_linear'
    indexWeightStr = 'relative_routeValues'
    fid = open("routeDifferenceInfo.csv","w+")

    label = "routeA-routeB"
    print(label)
    recordRouteDifference = compareRoutesByClass(routeA,routeB,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)

    label = "routeA-routeBPos"
    print(label)
    recordRouteDifference = compareRoutesByClass(routeA,routeBPos,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)

    label = "routeA-routeBNeg"
    print(label)
    recordRouteDifference = compareRoutesByClass(routeA,routeBNeg,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)

    label = "routeB-routeAPos"
    print(label)
    recordRouteDifference = compareRoutesByClass(routeB,routeAPos,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)

    label = "routeB-routeANeg"
    print(label)
    recordRouteDifference = compareRoutesByClass(routeB,routeANeg,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)

    label = "routeAPos-routeBPos"
    print(label)
    recordRouteDifference = compareRoutesByClass(routeAPos,routeBPos,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)
    label = "routeANeg-routeBNeg"
    print(label)
    recordRouteDifference = compareRoutesByClass(routeANeg,routeBNeg,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)
    label = "routeAPos-routeBNeg"
    print(label)
    recordRouteDifference = compareRoutesByClass(routeAPos,routeBNeg,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)
    label = "routeANeg-routeBPos"
    print(label)
    recordRouteDifference = compareRoutesByClass(routeANeg,routeBPos,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)
    label = "routeAPos-routeANeg"
    print(label)
    recordRouteDifference = compareRoutesByClass(routeAPos,routeANeg,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)
    label = "routeBPos-routeBNeg"
    print(label)
    recordRouteDifference = compareRoutesByClass(routeBPos,routeBNeg,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)








