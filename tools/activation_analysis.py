#!/usr/bin/env python2

import matplotlib
matplotlib.use('Agg')

import _init_paths

from fresh_config import cfg,cfgForCaching,get_config_field_update_template,update_config
from cache.one_level_cache import Cache
from fresh_util import load_data,print_measure_results,aggregate_results
from fresh_cluster import cluster_data,compute_clustering_statistics
from fresh_svm import compute_separability
from fresh_plot import plot_aggregate_scores_vs_clusters,plot_measure_list,tile_by_comboid

import numpy as np
np.set_printoptions(precision=3)
from pprint import pprint as pp


def find_optimal_k(train_data,test_data,kmeans_search_list):
    """
    [limitation: just a point] [cluster based on training data only]
    input:
       - clustering algorithm (always kmeans)
       - training data
       - testing data
       - reform fuction (reform)
       - evalution criterion (EC)
       - eval_function
       - reform function
       - pick_optimal_from_measures
    output:
      - optimal k
    """
    train_measures,test_measures = {'cluster':[],'separability':[]},{'cluster':[],'separability':[]}
    update_template = get_config_field_update_template(0,'density_estimation','nClusters')
    print(cfg)
    for k in kmeans_search_list:
        update_template.density_estimation.nClusters = k
        update_config(cfg,update_template)
        clusters = cluster_data(train_data)
        train_measure,test_measure = compute_cluster_measures(train_data,test_data,clusters)
        append_measures(train_measures,train_measure)
        append_measures(test_measures,test_measure)
    # pp(train_measures)
    # pp(test_measures)
    k_optim = pick_k_optimal(train_measures,test_measures,kmeans_search_list)
    return k_optim,train_measures,test_measures

def append_measures(measure_dict,to_add):
    for key in to_add.keys():
        measure_dict[key].append(to_add[key])

def pick_k_optimal(train_measures,test_measures,kmeans_search_list):
    number_of_k = len(kmeans_search_list)
    trainScores = aggregate_results(train_measures,number_of_k)
    testScores = aggregate_results(test_measures,number_of_k)

    # print results
    # print("train scores")
    # print_measure_results(train_measures,kmeans_search_list)
    # print("test scores")
    # print_measure_results(test_measures,kmeans_search_list)

    # print("train aggregate scores")
    # for index,k_index in enumerate(kmeans_search_list):
    #     print("@ {}: {}".format(k_index,trainScores[index]))

    # print("test aggregate scores")
    # for index,k_index in enumerate(kmeans_search_list):
    #     print("@ {}: {}".format(k_index,testScores[index]))

    # print("added scores")
    # addedScores = []
    # for index,k_index in enumerate(kmeans_search_list):
    #     addedScore = trainScores[index] + testScores[index]
    #     addedScores.append(addedScore)
    #     print("@ {}: {}".format(k_index,addedScore))

    # plot results
    plot_aggregate_scores_vs_clusters(trainScores,testScores,kmeans_search_list)
    plot_measure_list(train_measures,test_measures,kmeans_search_list)
    tile_by_comboid(cfg.comboInfo)
    
    return 0#kmeans_search_list[np.argmax(addedScores)]

def compute_cluster_measures(train_data,test_data,clusters):
   """
   - ec
      - clustering statistics 
        - [internal] silhouette coefficient
	- [external] purity (among data_classes) {correlates with interpretation}
	- [external] purity (among correct/incorrect) {correlates closes with "how good is my cluster"}
      - separability 
	- svm
   """
   measureCache = Cache("measure_cache.pkl",cfgForCaching,'measures')
   sets_measures = measureCache.load()
   if measureCache.is_valid: return sets_measures[0],sets_measures[1]

   if cfg.verbose: print("--> computing cluster statistics <--")
   train_clustering_statistics,test_clustering_statistics = compute_clustering_statistics(train_data,test_data,clusters)
   if cfg.verbose: print("--> computing separability <--")
   train_separability,test_separability = compute_separability(train_data,test_data,clusters)

   train_measures = {'cluster':train_clustering_statistics,'separability':train_separability}
   test_measures = {'cluster':test_clustering_statistics,'separability':test_separability}

   measureCache.save([train_measures,test_measures])
   return train_measures,test_measures


def compute_kmean_measures_FOR_model_data_combos():
    print(cfgForCaching)
    modelInfo = cfg.modelInfo
    train_imdb_name = cfg.data.train_imdb
    test_imdb_name = cfg.data.test_imdb
    density_estimation_algorithm = cfg.density_estimation.algorithm
    kmeans_search_list = cfg.density_estimation.search
    comboInfo = cfg.comboInfo
    layerList = cfg.layerList
    
    # load data
    train_data = load_data(train_imdb_name,layerList,modelInfo)
    test_data = load_data(test_imdb_name,layerList,modelInfo)

    number_gates_by_correctness_train = countUniqueActivationGates(train_data)
    number_gates_by_correctness_test = countUniqueActivationGates(test_data)
    print(number_gates_by_class_train)
    print(number_gates_by_class_test)

    # train_data = None
    # test_data = None

    # find the "best" k
    k_optim,train_msr,test_msr = find_optimal_k(train_data,test_data,kmeans_search_list)

    print("optimal k: {}".format(k_optim))
    return train_msr,test_msr

if __name__ == "__main__":

    """
    where is the data from?
    - model for activation values
    - (train data) passed throught activation values
    - (test data) passed throught activation values
    """
    compute_kmean_measures_FOR_model_data_combos()
