#!/usr/bin/env python2
import os,sys,re,yaml,subprocess,pickle,argparse
from pprint import pprint as pp
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets.ds_utils import loadEvaluationRecordsFromPath
from datasets.factory import get_repo_imdb
from core.config import cfg, cfg_from_file
from core.train import get_training_roidb
from sklearn import svm
from sklearn import cluster as sk_cluster

# LAYERS = ["conv1","ip1","cls_score"]
LAYERS = ["conv1","conv2","ip1","cls_score"]
#LAYERS = ["ip1","cls_score"]
# LAYERS = ["conv1","cls_score"]
# LAYERS = ["conv1"]


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

def aggregateKmeansCluster(kmeans,data):
    distance_data = kmeans.transform(data)
    distanceIndex = np.argsort(-distance_data)
    distance_data = distance_data[distanceIndex]
    print(distance_data[:10])
    print(distance_data[-10:])

def routeFromActivationsSet(activations,layerName,cacheStr=''):

    cluster = None
    useDBSCAN = False
    useKMeans = True
    clusterStr = ''
    useCluster = useDBSCAN | useKMeans
    nClusters_kmeans = 100
    # nSample = 500
    # dataIndex = np.random.permutation(activations.shape[0])[:nSample]
    # clusterData = activations.reshape((activations.shape[0],-1))[dataIndex,:]
    # #print("clusterData.shape: {}".format
    clusterData = activations.reshape((activations.shape[0],-1))

    if useCluster and clusterStr != '':
        netName = cfg.netName
        if cacheStr != '': clusterStr = "{}_{}".format(clusterStr,cacheStr)
        else: clusterStr = clusterStr
        cluster = loadClusterCache(clusterStr,netName,layerName)

    if useDBSCAN and cluster is None:
        # min_samples (default = 5)
        # (eps,#clusters) = { (1000,1,"-1"), (10000,1,"-1"), (11250,1,"-1"),
        # (12500,3,"-1,0,1"), (25000,2,"-1,0"), (50000,1,"0"), (100000,1,"0") }
        # min_samples = 3
        # (eps,#clusters) = { (1000,1,"-1"), (10000,1,"-1"), (11250,1,"-1"),
        # (12500,3,"-1,0,1"), (25000,2,"-1,0"), (50000,1,"0"), (100000,1,"0") }
        clusterStr = 'dbscan'
        print("{} in progress".format(clusterStr))
        dbscan = sk_cluster.DBSCAN(min_samples = 2,eps=12500).fit(clusterData)
        cluster = dbscan
        print(len(dbscan.labels_))
        print(dbscan.labels_.shape)
        print(dbscan.labels_)
        groups = np.unique(dbscan.labels_)
        nGroups = len(groups)
        print("number of groups: {}".format(nGroups))
        print("groups: {}".format(np.unique(dbscan.labels_)))
        # if nGroups == 1:
        #     sys.exit()

    if useKMeans and cluster is None:
        clusterStr = 'kmeans'
        print("{} in progress".format(clusterStr))
        kmeans = sk_cluster.KMeans(n_clusters=nClusters_kmeans).fit(clusterData)
        cluster = kmeans
        print(kmeans.labels_.shape)
        print(kmeans.cluster_centers_.shape)
        values = kmeans.cluster_centers_
        route = np.argsort(-np.abs(values))
        # aggregateKmeansCluster(kmeans,clusterData)


    if useCluster:
        if cacheStr != '': clusterStr = "{}_{}".format(clusterStr,cacheStr)
        else: clusterStr = clusterStr
        netName = cfg.netName
        saveClusterCache(cluster,clusterStr,netName,layerName)

    # conv layers are ordered by channel; weights are along channel, not across
    if 'conv' in layerName and useCluster is False:
        mean = np.mean(activations,axis=(0,1)).ravel()
        route = np.argsort(-np.abs(mean))
        values = mean.ravel()
        # mean = np.mean(activations,axis=0)
        # # std = np.std(activations,axis=0)
        # mask = (np.abs(mean) > thresh)
        # note: we can use the values of feature maps as indices to determine....
        # ---- what?
        # count how many are "on"
        # count = np.sum(mask,axis=0).ravel()
        # route = np.argsort(-count)
        # full connected layers individual since each node gets its own weight
    elif useCluster is False:
        mean = np.mean(activations,axis=0)
        route = np.argsort(-np.abs(mean))
        values = mean.ravel()
        # print(mean)
        # # std = np.std(activations,axis=0)
        # mask = (np.abs(mean) > thresh)
        # route = np.argsort(-mask)
        # print("*********")
        # print(mean.shape)
        # print(route.shape)
        # if len(route.shape) == 1:
        #     print(mean)
        #     print(route)
    activeDict = createRouteDict(values,route,cluster=cluster)
    return activeDict

def routingStatisticsByClass(clsDict,thresh=0.01,cacheStr=''):
    if clsDict is None: return None
    routeDict = dict.fromkeys(clsDict.keys())
    # 2nd compute (mean,std) of activations; order the activations based on average
    for layerName,clsLayerDict in clsDict.items():
        routeDict[layerName] = {}
        for clsName,clsLayerActivations_l in clsLayerDict.items():
            if clsName != "cat": continue # shortcut for computation
            clsLayerActivations = np.concatenate(clsLayerActivations_l,axis=0)
            if cacheStr != '': cacheStr = '{}_{}'.format(cacheStr,clsName)
            else: cacheStr = clsName
            routdClsLayerDict = routeFromActivationsSet(clsLayerActivations,layerName,cacheStr)
            routeDict[layerName][clsName] = routdClsLayerDict
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
    routeDict = routingStatisticsByClass(clsDict,thresh=threshold,cacheStr='all')
    print("routeDictPos")
    routeDictPos = routingStatisticsByClass(clsDictPos,thresh=threshold,cacheStr='pos')
    print("routeDictNeg")
    routeDictNeg = routingStatisticsByClass(clsDictNeg,thresh=threshold,cacheStr='neg')
    return routeDict,routeDictPos,routeDictNeg

def compareRoutes(routePrimary,routeSecondary,indexWeightStr=None):
    if routePrimary['cluster'] is not None:
        diff = compareRoutesWithCluster(routePrimary,routeSecondary,indexWeightStr = indexWeightStr)
    else:
        diff =  spearmanFootruleDistance(routePrimary,routeSecondary,indexWeightStr = indexWeightStr)        
    return diff

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
        diff += compareRoutes(centroidRouteA,centroidRouteB,indexWeightStr=indexWeightStr)

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

def addRouteEntry(fid,routeDifference,label):
    totalDiff = computeTotalRouteDifference(routeDifference)            
    fid.write("{},{}\n".format(label,totalDiff))

def saveRouteInformation(route,infixStr):
    print("saving routing information")

def createRouteDict(routeValues,routeIndex,cluster=None):
    routeDict = {}
    routeDict['values'] = routeValues
    routeDict['index'] = routeIndex
    routeDict['cluster'] = cluster
    return routeDict

def activationValuesToSVMFormat(avDict,records,layerOrder,referenceRoute,datasetSize=None,clsName=None,imdb=None):
    print("[activationValuesToSVMFormat]: starting")
    imageIdOrder = avDict[layerOrder[0]].keys()
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
    for layerName in layerOrder:
        avLayerDict = avDict[layerName]
        #routeValues,routeIndex = aggregateRouteLayerValuesOverCls(referenceRoute[layerName],clsName)
        routeLayerRef = referenceRoute[layerName][clsName]
        dataLayer = [None for _ in imageIdOrder]
        for index,image_id in enumerate(imageIdOrder):
            imageLayerActivations = avLayerDict[image_id].ravel()
            imageLayerIndex = np.argsort(-imageLayerActivations)
            routeImageDict = createRouteDict(imageLayerActivations[np.newaxis,:],
                                             imageLayerIndex[np.newaxis,:])
            #routeImageDict = routeFromActivationsSet(imageLayerActivations,layerName)
            difference = compareRoutes(routeLayerRef,routeImageDict,\
                                       indexWeightStr='relative_routeValues')
            dataLayer[index] = difference
            #dataLayer[index] = layerActivations.ravel()[routeIndex]
        dataLayer = np.array(dataLayer)
        if len(dataLayer.shape) == 1:
            dataLayer = dataLayer[:,np.newaxis]
        data.append(dataLayer)
    data = np.concatenate(data,axis=1)
    return data,labels,dataCls

def aggregateRouteLayerValuesOverCls(routeLayer,clsName):
    if clsName is not None: return routeLayer[clsName]['values'],routeLayer[clsName]['index']
    ave = []
    for clsName,routeInfo in routeLayer.items():
        value = routeInfo['values']
        ave.append(value)
    ave = np.mean(np.stack(ave),axis=0)
    index = np.argsort(-np.abs(ave))
    return ave,index

def plotDataWithLabels(dataTrain,labelsTrain,vis=True):
    indexListA = [0,0,0,1,1,2]
    indexListB = [1,2,3,2,3,3]
    pltCount = 1
    for indexA,indexB in zip(indexListA,indexListB):
        subplotNumber = 610+pltCount
        plt.subplot(subplotNumber)
        plt.scatter(dataTrain[:,indexA],dataTrain[:,indexB],c=labelsTrain)
        plt.xlabel(LAYERS[indexA])
        plt.ylabel(LAYERS[indexB])
        pltCount += 1
    plt.savefig("plotDataWithLabels.png")
    # plt.show()
    # sys.exit()

def fitSvmModel(avDict_train,avDict_test,records_train,records_test,referenceRoute,clsName=None,imdbTrain=None,imdbTest=None):
    clsIndex = None
    if clsName is not None:
        clsIndex = imdbTrain.classes.index(clsName)
        print("[SVM] Route data w.r.t class name: {}".format(clsName))

    # fix layer ordering
    layerOrder = sorted(avDict_train.keys())

    # format data for svm model
    trainSize = 30000
    testSize = 10000
    dataTrain,labelsTrain,dataClsTrain = activationValuesToSVMFormat(avDict_train,records_train,
                                                                     layerOrder,referenceRoute,
                                                                     datasetSize = trainSize,
                                                                     clsName=clsName,imdb=imdbTrain)
    dataTest,labelsTest,dataClsTest = activationValuesToSVMFormat(avDict_test,records_test,
                                                                  layerOrder,referenceRoute,
                                                                  datasetSize = testSize,
                                                                  clsName=clsName,imdb=imdbTest)
    
    if clsIndex is not None:
        selIndex = np.where(dataClsTrain == clsIndex)[0]
        dataTrain = dataTrain[selIndex,:]
        labelsTrain = labelsTrain[selIndex]

        selIndex = np.where(dataClsTest == clsIndex)[0]
        dataTest = dataTest[selIndex,:]
        labelsTest = labelsTest[selIndex]

    plotDataWithLabels(dataTrain,labelsTrain,vis=True)

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


    # train svm model
    probability = False
    clf = svm.SVC(gamma='auto',probability=probability,kernel='linear',class_weight='balanced')
    clf.fit(dataTrain,labelsTrain)

    # testing data fit
    predict = clf.predict(dataTest)
    reportFitResults(predict,labelsTest,clsName,dataClsTest,probability,"test")
    
    # training data fit
    predict = clf.predict(dataTrain)
    reportFitResults(predict,labelsTrain,clsName,dataClsTrain,probability,"train")


def reportFitResults(predict,labels,clsName,clsLabels,probability,dsStr):
    print("------- RESULTS FOR {} DATASET ----------".format(dsStr))
    print("[SVM] predicted number of 0's: {}".format(np.sum(predict == 0)))
    print("[SVM] predicted number of 1's: {}".format(np.sum(predict == 1)))

    if probability:
        aveOfabs = np.mean(np.abs(predict - labels))
        print("accuracy: {}".format(aveOfabs))

        selIndex = labels == 1
        aveOfabs_pos = np.mean(np.abs(predict[selIndex] - labels[selIndex]))
        print("pos")
        print(aveOfabs_pos)

        selIndex = labels == 0
        aveOfabs_neg = np.mean(np.abs(predict[selIndex] - labels[selIndex]))
        print("neg")
        print(aveOfabs_neg)
    else:
        aveOfabs = np.mean(predict == labels)
        print("accuracy: {}".format(aveOfabs))

        selIndex = labels == 1
        aveOfabs_pos = np.sum(predict[selIndex]  == labels[selIndex])/float(labels[selIndex].size)
        print("pos overall: {}".format(aveOfabs_pos))

        selIndex = labels == 0
        aveOfabs_neg = np.sum(predict[selIndex]  == labels[selIndex])/float(labels[selIndex].size)
        print("neg overall: {}".format(aveOfabs_neg))

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

def saveClusterCache(clusterOutput,clusterStr,netName,layerName):
    dirName = "./data/routing_cache/{netName}".format(netName=netName)
    if not osp.exists(dirName):
        os.makedirs(dirName)
    fn = "{dirName}/{clusterStr}_{layerName}.pkl".format(dirName=dirName,clusterStr=clusterStr,layerName=layerName)
    print("saving cluster cache: {}".format(fn))
    with open(fn,'wb') as f:
        pickle.dump(clusterOutput,f)

def loadClusterCache(clusterStr,netName,layerName):
    dirName = "./data/routing_cache/{netName}".format(netName=netName)
    fn = "{dirName}/{clusterStr}_{layerName}.pkl".format(dirName=dirName,clusterStr=clusterStr,layerName=layerName)
    if not osp.exists(fn): return None
    print("loading cluster cache: {}".format(fn))
    with open(fn,'rb') as f:
        data = pickle.load(f)
    return data

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

    cfg.netName = netName

    imdbA,roidbA = get_roidb(imdbNameA)
    imdbB,roidbB = get_roidb(imdbNameB)

    recordsA = loadEvaluationRecordsFromPath(recordsPathA)
    recordsB = loadEvaluationRecordsFromPath(recordsPathB)

    plotRecordDistribution = False
    if plotRecordDistribution:
        recordDistributionPerClassPerSampleSize(recordsA,imdbA)
        recordDistributionPerClassPerSampleSize(recordsB,imdbB)
    
    
    dirPathA = osp.join("./output/activity_vectors/classification/",
                   "{}-{}".format(datasetName,imageSetA))
    dirPathB = osp.join("./output/activity_vectors/classification/",
                   "{}-{}".format(datasetName,imageSetB))
    avA = loadActivityVectors(dirPathA,LAYERS,netName)
    avB = loadActivityVectors(dirPathB,LAYERS,netName)
    


    routeA,routeAPos,routeANeg = routeActivityVectors(avA,imdbA,roidbA,records=recordsA)
    fitSvmModel(avA,avB,recordsA,recordsB,routeA,clsName='cat',imdbTrain=imdbA,imdbTest=imdbB)
    #fitSvmModel(avA,avB,recordsA,recordsB,routeA)
    sys.exit()


    routeB,routeBPos,routeBNeg = routeActivityVectors(avB,imdbB,roidbB,records=recordsB)

    saveRouteInfo = False
    if saveRouteInfo:
        saveRouteInformation(routeA,"route")
        saveRouteInformation(routeAPos,"routePos")
        saveRouteInformation(routeANeg,"routeNeg")

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








