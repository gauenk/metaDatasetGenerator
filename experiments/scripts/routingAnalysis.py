#!/usr/bin/env python2
import os,sys,re,yaml,subprocess,pickle,argparse
from pprint import pprint as pp
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from datasets.ds_utils import loadEvaluationRecordsFromPath
from datasets.factory import get_repo_imdb
from core.config import cfg, cfg_from_file
from core.train import get_training_roidb

# LAYERS = ["conv1","conv2","ip1","cls_score"]
LAYERS = ["conv1"]


def get_roidb(imdb_name):
    imdb = get_repo_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    roidb = get_training_roidb(imdb)
    return imdb, roidb

def readPickle(fn):
    print(fn)
    tmp = None
    with open(fn,'rb') as f:
        tmp =pickle.load(f)
    return tmp

def loadActivityVectors(dirPath,layerList,netName):
    avDict = {}
    for layer in layerList:
        fn = osp.join(dirPath,"{}_{}".format(layer,netName)+".pkl")
        avDict[layer] = readPickle(fn)
    return avDict

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test an Object Detection network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='required config file', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def aggregateActivationsByClass(avDict,imdb,roidb,records):
    clsDict = {}
    clsDictPos = {}
    clsDictNeg = {}
    # 1st aggregrate the activations to be within each class within each layer
    for layer,avLayerDict in avDict.items():
        print("Layer {}".format(layer))
        clsDict[layer] = dict.fromkeys(imdb.classes,None)
        clsDictPos[layer] = dict.fromkeys(imdb.classes,None)
        clsDictNeg[layer] = dict.fromkeys(imdb.classes,None)
        for clsName in clsDict[layer].keys():
            clsDict[layer][clsName] = []
            clsDictPos[layer][clsName] = []
            clsDictNeg[layer][clsName] = []
        for image_id,activation in avLayerDict.items():
            index = imdb.image_index.index(image_id)
            clsIndex = roidb[index]['gt_classes'][0]
            clsName = imdb.classes[clsIndex]
            clsDict[layer][clsName].append(activation)
            if records is not None:
                recordValue = records[image_id][0]
                if recordValue == 1:
                    clsDictPos[layer][clsName].append(activation)
                elif recordValue == 0:
                    clsDictNeg[layer][clsName].append(activation)
                else:
                    print("weirdness worth checking out.\
                    records gives {} at image_id {}".format(recordValue,image_id))
    if records is None:
        clsDictPos = None
        clsDictNeg = None
    return clsDict,clsDictPos,clsDictNeg

def routingStatisticsByClass(clsDict,thresh=0.01):
    if clsDict is None: return None
    routeDict = dict.fromkeys(clsDict.keys())
    # 2nd compute (mean,std) of activations; order the activations based on average
    for layer,clsLayerDict in clsDict.items():
        routeDict[layer] = {}
        for clsName,clsLayerActivations_l in clsLayerDict.items():
            print(len(clsLayerActivations_l))
            clsLayerActivations = np.concatenate(clsLayerActivations_l,axis=0)
            # conv layers are ordered by channel; weights are along channel, not across
            if 'conv' in layer:
                mean = np.mean(clsLayerActivations,axis=0)
                # std = np.std(clsLayerActivations,axis=0)
                mask = (np.abs(mean) > thresh)
                # note: we can use the values of feature maps as indices to determine....
                # ---- what?
                # count how many are "on"
                count = np.sum(mask,axis=0).ravel()
                route = np.argsort(-count)
            # full connected layers individual since each node gets its own weight
            else:
                mean = np.mean(clsLayerActivations,axis=0)
                route = np.argsort(-np.abs(mean))
                # print(mean)
                # # std = np.std(clsLayerActivations,axis=0)
                # mask = (np.abs(mean) > thresh)
                # route = np.argsort(-mask)
            # print("*********")
            # print(mean.shape)
            # print(route.shape)
            # if len(route.shape) == 1:
            #     print(mean)
            #     print(route)
            routeDict[layer][clsName] = {}
            routeDict[layer][clsName]['index'] = route
            routeDict[layer][clsName]['values'] = mean.ravel()
    return routeDict

def routeActivityVectors(avDict,imdb,roidb,records=None):
    """
    input:
    - avDict (a dictionary with activations; 
              -keys = layer name, values = dictionary 
              -keys = image_ids, values = activations for layer
    - imdb (the original dataset; we need the labels)
    - threshold value; we need a smart way to set this...
    output:
    - routeDict: a dictionary with 
              -keys = layer name, values = dictionary               
              -keys = class_name, values = dictionary
    "index": ordered list of integers representing the descending
    ordering of the activation values
    "values": the values at each index
    - *routeDict (correct only)
    - *routeDict (incorrect only)
    *Note: if "records" is passed in then the output is three dictionaries
    """
    threshold = 0.01
    clsDict,clsDictPos,clsDictNeg = aggregateActivationsByClass(avDict,imdb,roidb,records)
    print("routeDict")
    routeDict = routingStatisticsByClass(clsDict,thresh=threshold)
    print("routeDictPos")
    routeDictPos = routingStatisticsByClass(clsDictPos,thresh=threshold)
    print("routeDictNeg")
    routeDictNeg = routingStatisticsByClass(clsDictNeg,thresh=threshold)
    return routeDict,routeDictPos,routeDictNeg

def compareRoutes(routeDictA,routeDictB,indexWeightStr=None):
    recordRouteDifference = {}
    layers = routeDictA.keys()
    classes = routeDictA[layers[0]].keys()
    for layer in layers:
        recordRouteDifference[layer] = {}
        for cls in classes:
            routeIndexA = routeDictA[layer][cls]['index']
            routeValuesA = routeDictA[layer][cls]['values']
            routeIndexB = routeDictB[layer][cls]['index']
            routeValuesB = routeDictB[layer][cls]['values']
            difference = spearmanFootruleDistance(routeIndexA,routeIndexB,
                                                  routeValuesA,routeValuesB,
                                                  indexWeightStr = indexWeightStr)
            recordRouteDifference[layer][cls] = difference
    return recordRouteDifference

def spearmanFootruleDistance(routeIndexA,routeIndexB,\
                             routeValuesA,routeValuesB,indexWeightStr=None):
    indexWeight = getIndexWeight(indexWeightStr,routeIndexA)
    absDiff = np.abs(np.abs(routeValuesA[routeIndexA]) - np.abs(routeValuesB[routeIndexA]))
    return np.mean(absDiff * indexWeight)

def getIndexWeight(indexWeightStr,routeIndex,verbose=False):
    indexWeight = np.ones(routeIndex.size)
    if verbose: print("indexWeightStr is {}: default uniform".format(indexWeightStr))
    if indexWeight is None: return indexWeight
    elif indexWeightStr == 'descending_linear':
        indexWeight = np.arange(routeIndex.size)
        indexWeight = indexWeight[::-1] / float(routeIndex.size)
    return indexWeight

def computeTotalRouteDifference(routeDifference):
    totalDiff = 0
    for layer,clsRouteDiff in routeDifference.items():
        for cls,values in clsRouteDiff.items():
            totalDiff += values
    return totalDiff

def addRouteEntry(fid,routeDifference,label):
    totalDiff = computeTotalRouteDifference(routeDifference)            
    fid.write("{},{}\n".format(label,totalDiff))

def saveRouteInformation(routeA,"route")


if __name__ == "__main__":
    
    args = parse_args()
    # config file
    configFn = args.cfg_file
    print(configFn)
    cfg_from_file(configFn)

    # Not used yet, but important for comparison in the future. what else tell us this?
    # maybe we need the "records" to save this info as well.
    netName = "cifar_10_lenet5_yesImageNoise_noPrune_iter_100000"
    datasetName = "cifar_10"
    modelType = "lenet5"
    imageSetA = "train"
    imageSetB = "val"
    imdbNameA = "{}-{}-default".format(datasetName,imageSetA)
    imdbNameB = "{}-{}-default".format(datasetName,imageSetB)
    recordsPathA = "./output/{modelType}/tp_fn_records/{imdbName}/records_cls_{netName}.pkl"\
                  .format(modelType=modelType,imdbName=imdbNameA,netName=netName)
    recordsPathB = "./output/{modelType}/tp_fn_records/{imdbName}/records_cls_{netName}.pkl"\
                  .format(modelType=modelType,imdbName=imdbNameB,netName=netName)

    imdbA,roidbA = get_roidb(imdbNameA)
    imdbB,roidbB = get_roidb(imdbNameB)

    recordsA = loadEvaluationRecordsFromPath(recordsPathA)
    recordsB = loadEvaluationRecordsFromPath(recordsPathB)
    
    dirPathA = osp.join("./output/activity_vectors/classification/",
                   "{}-{}".format(datasetName,imageSetA))
    dirPathB = osp.join("./output/activity_vectors/classification/",
                   "{}-{}".format(datasetName,imageSetB))
    avA = loadActivityVectors(dirPathA,LAYERS,netName)
    avB = loadActivityVectors(dirPathB,LAYERS,netName)
    
    routeA,routeAPos,routeANeg = routeActivityVectors(avA,imdbA,roidbA,records=recordsA)
    routeB,routeBPos,routeBNeg = routeActivityVectors(avB,imdbB,roidbB,records=recordsB)

    saveRouteInfo = True
    if saveRouteInfo:
        saveRouteInformation(routeA,"route")
        saveRouteInformation(routeAPos,"routePos")
        saveRouteInformation(routeANeg,"routeNeg")

    indexWeightStr = None
    #indexWeightStr = 'descending_linear'
    fid = open("routeDifferenceInfo.csv","w+")

    label = "routeA-routeB"
    print(label)
    recordRouteDifference = compareRoutes(routeA,routeB,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)

    label = "routeA-routeBPos"
    print(label)
    recordRouteDifference = compareRoutes(routeA,routeBPos,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)

    label = "routeA-routeBNeg"
    print(label)
    recordRouteDifference = compareRoutes(routeA,routeBNeg,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)

    label = "routeB-routeAPos"
    print(label)
    recordRouteDifference = compareRoutes(routeB,routeAPos,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)

    label = "routeB-routeANeg"
    print(label)
    recordRouteDifference = compareRoutes(routeB,routeANeg,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)

    label = "routeAPos-routeBPos"
    print(label)
    recordRouteDifference = compareRoutes(routeAPos,routeBPos,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)
    label = "routeANeg-routeBNeg"
    print(label)
    recordRouteDifference = compareRoutes(routeANeg,routeBNeg,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)
    label = "routeAPos-routeBNeg"
    print(label)
    recordRouteDifference = compareRoutes(routeAPos,routeBNeg,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)
    label = "routeANeg-routeBPos"
    print(label)
    recordRouteDifference = compareRoutes(routeANeg,routeBPos,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)
    label = "routeAPos-routeANeg"
    print(label)
    recordRouteDifference = compareRoutes(routeAPos,routeANeg,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)
    label = "routeBPos-routeBNeg"
    print(label)
    recordRouteDifference = compareRoutes(routeBPos,routeBNeg,
                                          indexWeightStr=indexWeightStr)
    addRouteEntry(fid,recordRouteDifference,label)
    pp(recordRouteDifference)








