import os,sys,re
#from core.config import cfgRouting as cfg
import os.path as osp
import numpy as np
from core.routingConfig import cfg, getResultsBaseFilenameRouting
from utils.misc import get_roidb,readPickle,writePickle
from sklearn import cluster as sk_cluster

"""
Clustering methods for routing information
"""

def getClusterCacheFilename(clusterCacheStr):
    dirName = "./data/routing_cache/{modelName}".format(modelName=cfg.netInfo.modelName)
    if not osp.exists(dirName):
        os.makedirs(dirName)
    baseFn = getResultsBaseFilenameRouting()
    fn = "{dirName}/{baseFn}_{clusterCacheStr}.pkl".format(dirName=dirName,\
                                                                 baseFn=baseFn,\
                                                                 clusterCacheStr=clusterCacheStr)
    return fn

def saveClusterCache(clusterOutput,clusterParams,clusterCacheStr):
    fn = getClusterCacheFilename(clusterCacheStr)
    print("saving cluster cache: {}".format(fn))
    writePickle(fn,{"cluster":clusterOutput,"params":clusterParams})

def loadClusterCache(clusterCacheStr):
    fn = getClusterCacheFilename(clusterCacheStr)
    print("loading cluster cache: {}".format(fn))
    if not osp.exists(fn): return None
    data = readPickle(fn)
    return data['cluster']

def clusterDataToCreateRouteFromActicationsSet(data,params,clusterCacheStr):
    loadCluster = osp.exists(getClusterCacheFilename(clusterCacheStr))
    if loadCluster:
        cluster = loadClusterCache(clusterCacheStr)
        values = cluster.cluster_centers_
        index = np.argsort(-np.abs(values))
    else:
        cluster,values,index = createClusterDataToCreateRouteFromActicationsSet(data,params,clusterCacheStr)
    return cluster,values,index

def averageDataToCreateRouteFromActicationsSet(activations):
    # conv layers are ordered by channel; weights are along channel, not across
    if 'conv' in layerName:
        mean = np.mean(activations,axis=(0,1)).ravel()
        index = np.argsort(-np.abs(mean))
        values = mean.ravel()
    else:
        mean = np.mean(activations,axis=0)
        index = np.argsort(-np.abs(mean))
        values = mean.ravel()
    return cluster,values,index

def createClusterDataToCreateRouteFromActicationsSet(data,params,clusterCacheStr):
    clusterData = data.reshape((data.shape[0],-1))
    clusterType = cfg.routingAnalysisInfo.densityEstimation.clusterTypeStr
    clusterParams = params[clusterType]
    if clusterType == 'dbscan':
        cluster,values,index = clusterWithDBSCAN(clusterData,clusterParams)
    elif clusterType == 'kmeans':
        cluster,values,index = clusterWithKMeans(clusterData,clusterParams)
    else:
        print("ERROR: not known clustering method \_(._.)_\\")
    saveClusterCache(cluster,params,clusterCacheStr)
    return cluster,values,index

def aggregateKmeansCluster(kmeans,data):
    print("[aggregateKmeansCluster]: What does bureaucracy and me have in common? We dont do anything")
    distance_data = kmeans.transform(data)
    distanceIndex = np.argsort(-distance_data)
    distance_data = distance_data[distanceIndex]
    print(distance_data[:10])
    print(distance_data[-10:])

def clusterWithKMeans(data,clusterParams):
    printStartClusteringInformation(clusterParams)
    kmeans = sk_cluster.KMeans(n_clusters=clusterParams['nClusters']).fit(data)
    # aggregateKmeansCluster(kmeans,clusterData)
    values,index = routeValuesAndIndexFromKMeans(kmeans,clusterParams)
    return kmeans,values,index

def printStartClusteringInformation(clusterParams):
    clusterType = cfg.routingAnalysisInfo.densityEstimation.clusterTypeStr
    print("{} clustering in progress with parameters:".format(clusterType))
    for key,value in clusterParams.items():
        print("-> {}: {}".format(key,value))

def routeValuesAndIndexFromKMeans(cluster,params):
    values = cluster.cluster_centers_
    index = np.argsort(-np.abs(values))
    return values, index

def clusterWithDBSCAN(data,clusterParams):
    dbscan_min_samples = clusterParams['minSamples']
    dbscan_eps = clusterParams['eps']
    clusterStr = 'dbscan_{}_{}'.format(dbscan_min_samples,dbscan_eps)
    # min_samples (default = 5)
    # (eps,#clusters) = { (1000,1,"-1"), (10000,1,"-1"), (11250,1,"-1"),
    # (12500,3,"-1,0,1"), (25000,2,"-1,0"), (50000,1,"0"), (100000,1,"0") }
    # min_samples = 3
    # (eps,#clusters) = { (1000,1,"-1"), (10000,1,"-1"), (11250,1,"-1"),
    # (12500,3,"-1,0,1"), (25000,2,"-1,0"), (50000,1,"0"), (100000,1,"0") }
    print("{} in progress".format(clusterStr))
    dbscan = sk_cluster.DBSCAN(min_samples = dbscan_min_samples,
                               eps=dbscan_eps).fit(data)
    # print(len(dbscan.labels_))
    # print(dbscan.labels_.shape)
    # print(dbscan.labels_)
    groups = np.unique(dbscan.labels_)
    nGroups = len(groups)
    print("number of groups: {}".format(nGroups))
    print("groups: {}".format(np.unique(dbscan.labels_)))
    # if nGroups == 1:
    #     sys.exit()
    values, index = routeValuesAndIndexFromDBSCAN(dbscan,clusterParams)
    return dbscan,values,index

def routeValuesAndIndexFromDBSCAN(cluster,params):
    print("DBSCAN: idk how to get values and index yet....")
    values = None
    index = None
    return values, index

