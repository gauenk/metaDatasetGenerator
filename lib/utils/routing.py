import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os,sys,re
import os.path as osp
from sklearn import cluster as sk_cluster
from utils.cluster import clusterDataToCreateRouteFromActicationsSet,averageDataToCreateRouteFromActicationsSet
from core.routingConfig import cfg, imdbFromDatasetDict, createDensityEstimationCacheStrID, getClassificationExperimentResultsTxtFilenameRouting, getResultsBaseFilenameRouting, unpackClassificationInformation, startTextFile


def compareRoutes(routePrimary,routeSecondary,indexWeightStr=None):
    if routePrimary['cluster'] is not None:
        diff = compareRoutesWithCluster(routePrimary,routeSecondary,indexWeightStr = indexWeightStr)
    else:
        diff =  spearmanFootruleDistance(routePrimary,routeSecondary,indexWeightStr = indexWeightStr)        
    return diff

def compareRoutesByClass(routeDictA,routeDictB,indexWeightStr=None):
    recordRouteDifference = {}
    layers = routeDictA.keys()
    classes = routeDictA[layers[0]].keys()
    for layer in layers:
        recordRouteDifference[layer] = {}
        for cls in classes:
            difference = compareRoutes(routeDictA[layer][cls],routeDictB[layer][cls],indexWeightStr=indexWeightStr)
            recordRouteDifference[layer][cls] = difference
    return recordRouteDifference

def spearmanFootruleDistance(routePrimary,routeSecondary,indexWeightStr=None):
    routeIndexA = np.squeeze(routePrimary['index'])
    routeValuesA = np.squeeze(routePrimary['values'])
    routeIndexB = np.squeeze(routeSecondary['index'])
    routeValuesB = np.squeeze(routeSecondary['values'])
    indexWeight = getIndexWeight(indexWeightStr,routeIndexA,routeValuesA)
    # print(routeValuesA.shape,routeIndexA.shape)
    # print(routeValuesB.shape,routeIndexB.shape)
    # relativeValuesA = np.abs(routeValuesA)/float(np.sum(routeValuesA))
    # relativeValuesB = np.abs(routeValuesB)/float(np.sum(routeValuesB))
    absDiff = np.abs(np.abs(routeValuesA[routeIndexA]) - np.abs(routeValuesB[routeIndexA]))
    return np.mean(absDiff * indexWeight)

def getIndexWeight(indexWeightStr,routeIndex,routeValues,verbose=False):
    #print(routeIndex.shape,routeIndex.size)
    indexWeight = np.ones(routeIndex.size)/(1. * routeIndex.size)
    if verbose: print("indexWeightStr is {}: default uniform".format(indexWeightStr))
    if indexWeight is None: return indexWeight
    elif indexWeightStr == 'descending_linear':
        indexWeight = np.arange(routeIndex.size)
        indexWeight = indexWeight[::-1] / float(routeIndex.size)
    elif indexWeightStr == 'relative_routeValues':
        indexWeight = np.abs(routeValues)/float(np.sum(np.abs(routeValues)))
        indexWeight = indexWeight[routeIndex]
    return indexWeight

def computeTotalRouteDifference(routeDifference):
    totalDiff = 0
    for layer,clsRouteDiff in routeDifference.items():
        for cls,values in clsRouteDiff.items():
            totalDiff += values
    return totalDiff

"""
Current: compare one references (one comboID result)
chang to: use the difference from multiple comboID's
"""

def compareRoutesWithCluster(routePrimary,routeSecondary,indexWeightStr=None):
    routeIndexA,routeValuesA,routeClusterA = routePrimary['index'],routePrimary['values'],routePrimary['cluster']
    routeIndexB,routeValuesB = routeSecondary['index'],routeSecondary['values']

    # ---- routePrimary ----
    # routeValues; an average over clusters in node (the centroids)
    # .size = (# of clusters,# of activations per sample)
    # routeIndex; a nonincreasing order for each index of the average activation across all samples
    # .size = (# of clusters,# of activations per sample)
    # routeCluster; the cluster method object after ".fit" is run with a ".predict" method for new data.

    # ---- routeSecondary ----
    # routeValues;
    #    (a) an average over clusters in node (the sample)
    #    (b) the value of a sample (from an image_id; think *online* evaluation)
    # .size = (# of clusters/samples,# of activation per sample)
    # routeIndex; a nonincreasing order for each index of ...
    #    (a) the average activation for all samples
    #    (b) a single sample (again, think *online* evaluation)
    # .size = (# of clusters/samples,# of activation per sample)
    
    # assign each centroid in route B to a centroid in route A
    routeBCentroidsLabelsWrtA = routeClusterA.predict(routeValuesB)
    
    # difference for each cluster
    diff = 0
    routeDiff = [0 for _ in routeClusterA.cluster_centers_]
    for centroidLabel in np.unique(routeBCentroidsLabelsWrtA):
        # print(centroidLabel)
        if centroidLabel == -1:
            print("ERROR: we shouldn't see a centroid label of -1")
            sys.exit()
        centroidIndexA = centroidLabel
        centroidRouteA = createRouteDict(routeValuesA[centroidIndexA,:],routeIndexA[centroidIndexA,:])

        centroidIndexB = np.where(routeBCentroidsLabelsWrtA == centroidLabel)[0]
        # skip nodes not assigned to cluster;
        # but this shouldn't happend since we iterate over b's centoid labels
        if len(centroidIndexB) == 0:
            print("ERROR: We shouldn't see a centroid label for which B is not assigned.")
            sys.exit()
        centroidRouteB = createRouteDict(routeValuesB[centroidIndexB,:],routeIndexB[centroidIndexB,:])
        diffCurrent = compareRoutes(centroidRouteA,centroidRouteB,indexWeightStr=indexWeightStr)
        routeDiff[centroidLabel] = 1
        diff += diffCurrent

    routeDiff.append(diff)
    # return diff
    routeDiff = np.array(routeDiff)
    return routeDiff

def activationValuesToSVMFormat(info):
    avDict,records,layerOrder,referenceRoute,datasetSize,clsName,imdb = unpackClassificationInformation(info)
    print("[activationValuesToSVMFormat]: starting")
    imageIdOrder = sorted(avDict[layerOrder[0]].keys())
    if datasetSize is not None:
        imageIdOrder = imageIdOrder[:datasetSize]

    labels = [None for _ in imageIdOrder]
    for index,image_id in enumerate(imageIdOrder):
        labels[index] = records[image_id][0]
    labels = np.array(labels)

    dataCls = None
    if imdb is not None:
        roidb = imdb.roidb
        dataCls = [None for _ in imageIdOrder]
        for index,image_id in enumerate(imageIdOrder):
            roidbIndex = imdb.image_index.index(image_id)
            dataCls[index] = roidb[roidbIndex]['gt_classes'][0]
        dataCls = np.array(dataCls)

    data = []
    indexWeightStr = cfg.clsExperimentInfo.referenceRoute.indexWeightStr
    comboInfo = cfg.routingAnalysisInfo.comboInfo
    ## avdict only has each layer name so we need to cat them together to get the 
    for comboID,comboParameters in comboInfo.items():
        #routeValues,routeIndex =\
        # aggregateRouteLayerValuesOverCls(referenceRoute[layerName],clsName)
        routeLayerRef = referenceRoute[comboID][clsName]
        dataLayer = [None for _ in imageIdOrder]
        for index,image_id in enumerate(imageIdOrder):
            imageLayerActivations = extractActivationValues(avDict,comboID,image_id)
            imageLayerIndex = np.argsort(-imageLayerActivations)
            routeImageDict = createRouteDict(imageLayerActivations[np.newaxis,:],
                                             imageLayerIndex[np.newaxis,:])
            #routeImageDict = routeFromActivationsSet(imageLayerActivations,layerName)
            difference = compareRoutes(routeLayerRef,routeImageDict,\
                                       indexWeightStr=indexWeightStr)
            dataLayer[index] = difference
            #dataLayer[index] = layerActivations.ravel()[routeIndex]
        dataLayer = np.array(dataLayer)
        if len(dataLayer.shape) == 1:
            dataLayer = dataLayer[:,np.newaxis]
        data.append(dataLayer)
    data = np.concatenate(data,axis=1)
    return data,labels,dataCls

def extractActivationValues(avDict,comboID,image_id):
    comboLayerNames = comboID.split("-")
    orderedLayerNames = cfg.routingAnalysisInfo.layers
    avVector = []
    for layerName in orderedLayerNames:
        if layerName not in comboLayerNames: continue
        avVector.extend(avDict[layerName][image_id].ravel())
    avVector = np.array(avVector).ravel()
    return avVector
    
def aggregateActiationsOfLayerByRecordIntoComboDicts(allCombo,allComboList,posCombo,posComboList,negCombo,negComboList):
    if cfg.verbose:
        print("[aggregateActiationsOfLayerByRecordIntoComboDicts]: len(allComboList): {}".format(len(allComboList)))
    allCombo.append(allComboList)
    if len(posComboList) > 0: posCombo.append(posComboList)
    if len(negComboList) > 0: negCombo.append(negComboList)
    
def initComboClassDictionary(comboParameters,classes):
    comboDict = {}
    for className in classes: comboDict[className] = []
    return comboDict

def aggregateActiationsByRecordIntoComboSplits(recordValue,activation,allCombo,posCombo,negCombo,imageIndexID):
    allCombo.extend(activation)
    if recordValue == 1:
        posCombo.extend(activation)
    elif recordValue == 0:
        negCombo.extend(activation)
    else:
        print("weirdness worth checking out.\
        records gives {} at imageIndexID {}".format(recordValue,imageIndexID))

def getClassName(imdb,roidb,imageID):
    index = imdb.image_index.index(imageID)
    clsIndex = roidb[index]['gt_classes'][0]
    clsName = imdb.classes[clsIndex]
    return clsName

def subsampleDictionary(pyDict,subsampleKeys):
    # retains the order of the subsample list
    subsampleDict = {subsampleKey:pyDict[subsampleKey] for key in subsampleKeys}
    return subsampleDict
    

def routeFromActivationsSet(activations,densityEstimationParameters,clusterCacheStr):
    densityEstimationType = cfg.routingAnalysisInfo.densityEstimation.typeStr
    if densityEstimationType == 'cluster':
        cluster,values,index = clusterDataToCreateRouteFromActicationsSet(activations,densityEstimationParameters[densityEstimationType],clusterCacheStr)
    else:
        cluster,values,index = averageDataToCreateRouteFromActicationsSet(activations,densityEstimationParameters[densityEstimationType],clusterCacheStr)
    activeDict = createRouteDict(values,index,cluster=cluster)
    return activeDict

def routingStatisticsByClass(combos,imdb,recordIncludeTypeStr,thresh=0.01):
    densityEstimationTypeConfig = cfg.routingAnalysisInfo.densityEstimation.typeConfig
    if combos is None: return None
    routeDict = {}
    for comboID,clsComboDict in combos.items():
        routeDict[comboID] = {}
        if cfg.routingAnalysisInfo.densityEstimation.classSeparate:
            for clsName in imdb.classes:
                if skipSampleForComputation(clsName): continue # shortcut for computation
                densityEstimationCacheStrID = createDensityEstimationCacheStrID(comboID,recordIncludeTypeStr,imdb,clsName)
                print(densityEstimationCacheStrID)
                routeClsComboDict = routeFromActivationsSet(combos[comboID][clsName],densityEstimationTypeConfig[clsName],densityEstimationCacheStrID)
                routeDict[comboID][clsName] = routeClsComboDict
        else:
            msg = "implement aggregatation *not* by class for routing information here"
            raise NotImplementedError(msg)

    return routeDict

def skipSampleForComputation(clsName):
    return cfg.loadOnlyClsStr is not None and cfg.loadOnlyClsStr != clsName    

def routingStatisticsByClassSeparateLayers(clsDict,imdb,recordIncludeTypeStr,thresh=0.01):
    densityEstimationTypeConfig = cfg.routingAnalysisInfo.densityEstimation.typeConfig
    if clsDict is None: return None
    routeDict = dict.fromkeys(clsDict.keys())
    # 2nd compute (mean,std) of activations; order the activations based on average
    for layerName,clsLayerDict in clsDict.items():
        routeDict[layerName] = {}
        for clsName,clsLayerActivations in clsLayerDict.items():
            if skipSampleForComputation(clsName): continue # shortcut for computation
            densityEstimationCacheStrID = createDensityEstimationCacheStrID(comboID,recordIncludeTypeStr,imdb,clsName)
            routeClsLayerDict,updateDensityEsimtaionConfig = routeFromActivationsSet(clsLayerActivations,densityEstimationTypeConfig[clsName],densityEstimationCacheStrID)
            if updateDensityEsimtaionConfig is not None:
                densityEstimationTypeConfig[clsName] = updateDensityEsimtaionConfig
            routeDict[layerName][clsName] = routeClsLayerDict
    return routeDict

def routeActivityVectorsByClass(avDict,imdb,roidb,numSamples=-1,records=None,comboType=None):
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
    comboAll,comboAllLabels,comboPos,comboNeg = aggregateActivations(avDict,imdb,roidb,records,numSamples)
    print("routeDict")
    routeDict = routingStatisticsByClass(clsDict,imdb,'all',thresh=threshold)
    print("routeDictPos")
    routeDictPos = routingStatisticsByClass(clsDictPos,imdb,'pos',thresh=threshold)
    return routeDict,routeDictPos,None
    # print("routeDictNeg")
    # routeDictNeg = routingStatisticsByClass(clsDictNeg,'neg',thresh=threshold)
    # return routeDict,routeDictPos,routeDictNeg
    saveRouteCacheBool = False
    if saveRouteCacheBool:
        saveRouteCache(routeA,"route")
        saveRouteCache(routeAPos,"routePos")
        saveRouteCache(routeANeg,"routeNeg")


def computeClustersByData():
    

def addRouteEntry(fid,routeDifference,label):
    totalDiff = computeTotalRouteDifference(routeDifference)            
    fid.write("{},{}\n".format(label,totalDiff))

def getRouteCacheFilename(layerName,dataSplitName,cacheStr):
    dirName = "./data/routing_cache/{netName}".format(netName=cfg.netName)
    if not osp.exists(dirName):
        os.makedirs(dirName)
    baseFn = getResultsBaseFilenameRouting()
    fn = "{dirName}/route_{baseFn}_{cacheStr}_{layerName}.pkl".format(dirName=dirName,\
                                                                      baseFn=baseFn,\
                                                                      cacheStr=cacheStr,\
                                                                      layerName=layerName)
    return fn

def saveRouteCache(route,infixStr):
    print("saving routing information")
    raise NotImplementedError

def loadRouteCache(route,infixStr):
    print("saving routing information")
    raise NotImplementedError

def createRouteDict(routeValues,routeIndex,cluster=None):
    routeDict = {}
    routeDict['values'] = routeValues
    routeDict['index'] = routeIndex
    routeDict['cluster'] = cluster
    return routeDict

def aggregateRouteLayerValuesOverCls(routeLayer,clsName):
    if clsName is not None: return routeLayer[clsName]['values'],routeLayer[clsName]['index']
    ave = []
    for clsName,routeInfo in routeLayer.items():
        value = routeInfo['values']
        ave.append(value)
    ave = np.mean(np.stack(ave),axis=0)
    index = np.argsort(-np.abs(ave))
    return ave,index

def reportClassificationExperimentResults(predict,labels,clsName,clsLabels,probability,splitWRTcls,dsSplitWRTRoute,imdb):
    dsName = imdb.name
    dsSplitOg = imdb._image_set
    dsConfig = 'default' # fix this
    size = predict.size
    fid = startTextFile('results',\
                        split_wrt_cls=splitWRTcls,imdb_wrt_cls=dsSplitWRTRoute,size=size)
    # fn = getClassificationExperimentResultsTxtFilenameRouting(dsName,dsSplitWRTRoute,dsConfig,dsSplitOg,dsConfig)
    # print(fn)
    # fid = open(fn,'w+')

    fid.write("------- RESULTS FOR {} DATASET ----------\n".format(dsSplitWRTRoute))
    fid.write("[SVM] predicted number of 0's: {}\n".format(np.sum(predict == 0)))
    fid.write("[SVM] predicted number of 1's: {}\n".format(np.sum(predict == 1)))

    if probability:
        aveOfabs = np.mean(np.abs(predict - labels))
        fid.write("accuracy: {}\n".format(aveOfabs))

        selIndex = labels == 1
        aveOfabs_pos = np.mean(np.abs(predict[selIndex] - labels[selIndex]))
        fid.write("pos overall: {}\n".format(aveOfabs_pos))

        selIndex = labels == 0
        aveOfabs_neg = np.mean(np.abs(predict[selIndex] - labels[selIndex]))
        fid.write("neg overall: {}\n".format(aveOfabs_neg))
    else:
        aveOfabs = np.mean(predict == labels)
        fid.write("accuracy: {}\n".format(aveOfabs))

        selIndex = labels == 1
        aveOfabs_pos = np.sum(predict[selIndex]  == labels[selIndex])/float(labels[selIndex].size)
        fid.write("pos overall: {}\n".format(aveOfabs_pos))

        selIndex = labels == 0
        aveOfabs_neg = np.sum(predict[selIndex]  == labels[selIndex])/float(labels[selIndex].size)
        fid.write("neg overall: {}\n".format(aveOfabs_neg))

        # if clsName is not None:
        #     selIndex = clsLabels == clsName
        #     print(predict[selIndex])
        #     aveOfabs_pos = np.sum(predict[selIndex]  == labels[selIndex])/float(labels[selIndex].size)
        #     print("accuracy [cls: {}]: {}".format(clsName,aveOfabs_pos))

        #     selIndex = [(labels == 1) & (clsLabels == clsName)]
        #     aveOfabs_pos = np.sum(predict[selIndex]  == labels[selIndex])/float(labels[selIndex].size)
        #     print("pos [cls: {}]: {}".format(clsName,aveOfabs_pos))

        #     selIndex = [(labels == 0) & (clsLabels == clsName)]
        #     aveOfabs_neg = np.sum(predict[selIndex]  == labels[selIndex])/float(labels[selIndex].size)
        #     print("neg [cls: {}]: {}".format(clsName,aveOfabs_neg))
            
    fid.close()



def recordDistributionPerClassPerSampleSize(records,imdb,vis=False):
    vis = True
    clsRecord = []
    imageIndexOrder = imdb.image_index
    roidb = imdb.roidb

    recordByCls = {}
    for clsIndex in range(len(imdb.classes)):
        recordByCls[clsIndex] = []

    for image_id in imageIndexOrder:
        roidbIndex = imdb.image_index.index(image_id)
        clsIndex = roidb[roidbIndex]['gt_classes'][0]
        recordByCls[clsIndex].append(records[image_id][0])

    for clsIndex in range(len(imdb.classes)):
        proportionPosRange = np.arange(len(recordByCls[clsIndex])) + 1
        proportionPos = np.cumsum(recordByCls[clsIndex]) / proportionPosRange.astype(np.float)
        plt.plot(proportionPosRange,proportionPos,label=imdb.classes[clsIndex])
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if vis:
        plt.show()
    else:
        plt.savefig(saveFn,bbox_extra_artists=(lgd,), bbox_inches='tight')

def clusterDataAggregate(dataTrain,layerSizes,index):
    indexLen = layerSizes[index] - 1
    # first N are indicators; final value is the difference at the only "1" value
    value = np.where(dataTrain[:,:indexLen-1] == 1)[1] * dataTrain[:,indexLen] 
    print(value)
    return value

def plotDataWithLabels(dataTrain,labelsTrain,layers,layerSizes,vis=True):
    # indexListA = [0]
    # indexListB = [0]
    indexListA = [0,0,0,1,1,2]
    indexListB = [1,2,3,2,3,3]
    pltCount = 1
    for indexA,indexB in zip(indexListA,indexListB):
        subplotNumber = 610+pltCount
        plt.subplot(subplotNumber)
        xData = clusterDataAggregate(dataTrain,layerSizes,indexA)
        yData = clusterDataAggregate(dataTrain,layerSizes,indexB)
        plt.scatter(xData,yData,c=labelsTrain)
        plt.xlabel(layers[indexA])
        plt.ylabel(layers[indexB])
        pltCount += 1
    if vis:
        plt.show()
    else:
        baseFn = getResultsBaseFilenameRouting()
        fn = "plotDataWithLabels_{}.png".format(baseFn)
        print("Saving plot at {}".format(fn))
        plt.savefig(fn)



