#!/usr/bin/env python2

import matplotlib
matplotlib.use('Agg')

import _init_paths

from fresh_plot import analyze_expfield_results
from fresh_util import Cache
from fresh_config import cfg,cfgForCaching,create_model_name,create_model_path,update_config,cfg_from_file
from fresh import compute_kmean_measures_FOR_model_data_combos

import argparse,sys
from copy import deepcopy
from easydict import EasyDict as edict
import numpy as np
np.set_printoptions(precision=3)
from pprint import pprint as pp



def generate_experiments_across_iterations():
    # architecture_list = ['lenet5','zf','vgg16','resNet_small']
    #train_set_list = ['cifar_10','mnist']
    architecture_list = ['lenet5']
    # train_set_list = ['cifar_10-train-default','mnist-train-default']
    train_set_list = ['mnist-train-default']
    test_set_dict ={
        'cifar_10':['cifar_10-train-default','cifar_10-val-default'],
        'mnist':['mnist-train-default','mnist-test-default'],
    }

    # iterations_list = np.arange(0,1000000+1,25000)[1:]
    iterations_list = [(idx+1)*20000 for idx in range(5)]
    # iterations_list = np.arange(0,1000000+1,500000)[1:]
    #prune_list = ['noPrune','yesPrune10','yesPrune100','yesPrune200']
    prune_list = [False]
    image_noise_list = [False]
    optim_list = ['adam']
    ds_aug_list = ['25-0','10-0',False]

    modelInfo_list = []
    expConfig = edict()
    expConfig.data = edict()
    expConfig.modelInfo = edict()
    modelInfo = expConfig.modelInfo
    for architecture in architecture_list:
        for train_set in train_set_list:
            train_set_name = train_set.split('-')[0]
            for test_set in test_set_dict[train_set_name]:
                for optim in optim_list:
                    for image_noise in image_noise_list:
                        for prune in prune_list:
                            for ds_aug in ds_aug_list:
                                for iterations in iterations_list:
                                    expConfig.data.train_imdb = train_set
                                    expConfig.data.test_imdb = test_set
                                    modelInfo.iterations = iterations
                                    modelInfo.train_set = train_set_name
                                    modelInfo.architecture = architecture
                                    modelInfo.prune = prune
                                    modelInfo.image_noise = image_noise
                                    modelInfo.optim = optim
                                    modelInfo.dataset_augmentation = ds_aug
                                    modelInfo.classFilter = False
                                    modelInfo.name = create_model_name(modelInfo)
                                    modelInfo.path = create_model_path(modelInfo)
                                    modelInfo_list.append(deepcopy(expConfig))
    return modelInfo_list


def get_generalization_error_information(train_msrs,test_msrs,exp_config):
    measure_name_dict = cfg.measure_name_dict
    marker_list = cfg.plot.marker_list
    unique_changes = get_unique_experiment_field_change(exp_configs)
    marker_index = dict.fromkeys(unique_changes,0)
    plot_dict = {}
    
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--cfg', dest='cfg_file',help='optional config file',type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    print("This file is provided for mangling the results from:")
    print("-> the optimal_k process from fresh.py")
    print("-> the separability of the original model on each dataset")

    args = parse_args()
    print('Called with args:')
    print(args)
    if args.cfg_file is not None:
        print("SET CFG FILE")
        cfg_from_file(args.cfg_file)

    # get experiment list
    experiment_configs = generate_experiments_across_iterations()

    # print experiment info
    # print(cfg)
    # pp(experiment_configs)

    # start cache
    fn = "experiment_results.pkl"
    # fn = 'output/fresh/pairs_no_conv1/experiment_results.pkl'
    expCache = Cache(fn,cfgForCaching,'results')

    # run experiments
    train_results = []
    test_results = []
    for experiment in experiment_configs:
        update_config(cfg,experiment)
        # caching: already computed? 
        expCache.config = cfg
        results = expCache.load()
        if expCache.is_valid: # load earlier results
            train_msr,test_msr = results[0],results[1]
        else: # run the experiment
            train_msr,test_msr = compute_kmean_measures_FOR_model_data_combos()
            expCache.save([train_msr,test_msr])
        # store results in list
        train_results.append(train_msr),test_results.append(test_msr)

    kmeans_search_list = cfg.density_estimation.search
    print(analyze_expfield_results(train_results,test_results,experiment_configs,'iterations',kmeans_search_list))
    
    # analyze results... later
    print(train_results)
    print(test_results)


        
