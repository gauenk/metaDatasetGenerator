from fres_config import cfgForCaching
from fresh_util import *

from sklearn import cluster as sk_cluster
from sklearn import metrics as sk_metrics

import numpy as np #only for "uniqu" debug

# ------------------------------------------------------------------------------------

"""

Creation of Clusters

"""

def cluster_data(data):
    clusterCache = Cache("cluster_cache.pkl",cfgForCaching,"cluster")
    clusters = clusterCache.load()
    if clusterCache.is_valid: return clusters

    clusters = {}
    for comboID,comboData in data.samples.items():
        clusters[comboID] = new_cluster(comboData,clusterParams)

    clusterCache.save(clusters)
    return clusters

def new_cluster(data,clusterParams):
    clusterAlgorithm = clusterParams.density_estimation.algorithm
    if clusterAlgorithm == 'dbscan':
        cluster = clusterWithDBSCAN(data,clusterParams)
    elif clusterAlgorithm == 'kmeans':
        cluster = clusterWithKMeans(data,clusterParams.density_estimation)
    else:
        print("ERROR: not known clustering method \_(._.)_\\")
    return cluster

def clusterWithKMeans(data,clusterParams):
    kmeans = sk_cluster.KMeans(n_clusters=clusterParams['nClusters']).fit(data)
    return kmeans

def clusterWithDBSCAN(data,clusterParams):
    raise NotImplementedError("*riiip* clusterWithDBSCAN not implemented.")


# ------------------------------------------------------------------------------------


"""

Evaluation of Clusters

"""

def compute_clustering_statistics(train_data,test_data,clusters):
    train_cluster_statistics = {}
    test_cluster_statistics = {}
    for comboID,comboData in train_data.samples.items():
        train_data.exp_samples = comboData
        train_cluster_statistics[comboID] = compute_set_clustering_statistics(train_data,
                                                                              clusters[comboID])
    for comboID,comboData in test_data.samples.items():
        test_data.exp_samples = comboData
        test_cluster_statistics[comboID] = compute_set_clustering_statistics(test_data,
                                                                              clusters[comboID])
    return train_cluster_statistics,test_cluster_statistics

def compute_set_clustering_statistics(data,clusters):
    if clusters.cluster_centers_.shape[0] == 1: return [0,0,0]
    samples = data.exp_samples
    ds_labels = data.ds_labels
    correct = data.correct
    labels = clusters.predict(samples)
    silhouette_score = sk_metrics.silhouette_score(samples,labels)
    homogeneity_score_by_data_class = sk_metrics.homogeneity_score(ds_labels,labels)
    homogeneity_score_by_correctness = sk_metrics.homogeneity_score(correct,labels)
    stats = [silhouette_score,homogeneity_score_by_data_class,homogeneity_score_by_correctness]
    return stats
        

    
