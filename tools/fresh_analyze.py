#!/usr/bin/env python

from easydict import EasyDict as edict
import os,sys,pickle
from os import path as osp

def inspect_config(config):
    print("--")
    for config_key,config_value in config.items():
        if type(config_value) is edict or type(config_value) is dict:
            print(config_key)
            for key,value in config_value.items():
                print(key,value)

measure_name_dict = {'cluster':['silhouette','homogeneity_ds_labels','homogeneity_correct'],
                     'separability':['separability_ds_labels','separability_ds_correct']}

if __name__ == "__main__":
    store_dir = './output/fresh'
    load_folder = 'pairs_no_conv1'
    if len(sys.argv) > 1: load_folder = sys.argv[1]
    exp_file = 'experiment_results.pkl'
    filename = '{}/{}/{}'.format(store_dir,load_folder,exp_file)
    with open(filename,'r') as f:
        results = pickle.load(f)
    print(results.keys())
    for uuid_str in results.keys():
        print(results[uuid_str]['config'].modelInfo.iterations)
        for index,item in enumerate(results[uuid_str]['data']['results']):
            # two items for 'train' and 'val' sets
            print(index,item.keys())
            print(len(item['cluster']))
            print(item['cluster'][0])
            print(item['separability'][0])
