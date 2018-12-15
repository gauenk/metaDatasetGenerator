#!/usr/bin/env python2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import _init_paths


from fresh_plot import init_plot,add_plot_data,mangle_plot_name
from fresh_config import cfg,create_model_name,create_model_path,reset_model_name
from fresh_util import Cache,transpose_measures,get_unique_strings,find_experiment_changes
from fresh import compute_kmean_measures_FOR_model_data_combos

from copy import deepcopy
from easydict import EasyDict as edict
import numpy as np
np.set_printoptions(precision=3)
from pprint import pprint as pp



def generate_experiments_across_iterations():
    # architecture_list = ['lenet5','zf','vgg16','resNet_small']
    #train_set_list = ['cifar_10','mnist']
    architecture_list = ['lenet5']
    train_set_list = ['cifar_10']
    # iterations_list = np.arange(0,1000000+1,25000)[1:]
    iterations_list = [(idx+1)*10000 for idx in range(10)]
    # iterations_list = np.arange(0,1000000+1,500000)[1:]
    #prune_list = ['noPrune','yesPrune10','yesPrune100','yesPrune200']
    prune_list = ['noPrune']
    image_noise_list = ['yesImageNoise']

    modelInfo_list = []
    expConfig = edict()
    expConfig.modelInfo = edict()
    modelInfo = expConfig.modelInfo
    for image_noise in image_noise_list:
        for prune in prune_list:
            for architecture in architecture_list:
                for train_set in train_set_list:
                    for iterations in iterations_list:
                        modelInfo.iterations = iterations
                        modelInfo.train_set = train_set
                        modelInfo.architecture = architecture
                        modelInfo.prune = prune
                        modelInfo.image_noise = image_noise
                        modelInfo.name = create_model_name(modelInfo)
                        modelInfo.path = create_model_path(modelInfo)
                        modelInfo_list.append(deepcopy(expConfig))
    return modelInfo_list

def update_config(input_cfg,experiment_config):
    for key,value in experiment_config.items():
        if key not in input_cfg.keys(): raise ValueError("key [{}] not in original configuration".format(key))
        if type(value) is edict: update_config(input_cfg[key],value)
        input_cfg[key] = value

def what_changed(new_cfg,old_cfg):
    for key,old_value in old_cfg.items():
        if key is 'name': continue
        new_value = new_cfg[key]
        if key not in new_cfg.keys(): raise ValueError("key [{}] not in original configuration".format(key))
        if type(old_value) is edict: 
            a,b = what_changed(new_value,old_value)
            if a is not None: return a,b
        if old_value != new_value: return key,new_value
    return None,None

def analyze_expfield_results(train_results,test_results,exp_configs,expfield_filter,kmeans_search_list):
    measure_name_dict = {'cluster':['silhouette','homogeneity_ds_labels','homogeneity_correct'],
                         'separability':['separability_ds_labels','separability_ds_correct']}
    change_field = find_experiment_changes(exp_configs)    
    fieldname_list = [field[0] for field in change_field]
    unique_changes = get_unique_strings(fieldname_list)
    marker_list = cfg.plot.marker_list
    marker_index = dict.fromkeys(unique_changes,0)
    print(marker_index)
    plot_dict = {}
    handle_dict = {}
    for field_index,exp_config in enumerate(exp_configs):
        # update to exp_config
        old_cfg = deepcopy(cfg)
        update_config(cfg,exp_config)
        fieldname,fieldvalue = what_changed(cfg,old_cfg)
        if fieldname != expfield_filter: continue
        reset_model_name()

        # update to exp_config
        train_msrs,test_msrs = train_results[field_index],test_results[field_index]
        print(fieldname)
        print(train_msrs)
        # if fieldname not in handle_dict.keys(): handle_dict[fieldname] = {}
        for measure_type in train_msrs.keys():
            msr_dict_type = measure_name_dict[measure_type]
            print("----------------------------------")
            print(msr_dict_type)
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print(train_msrs[measure_type])
            print(test_msrs[measure_type])
            tr_type_t,tr_msr_name_t = transpose_measures(train_msrs[measure_type],msr_dict_type)
            te_type_t,te_msr_name_t = transpose_measures(test_msrs[measure_type],msr_dict_type)
            print("************************************")
            # always in the same order
            for tr_msr,te_msr in zip(tr_msr_name_t,te_msr_name_t):
                assert tr_msr == te_msr,"train/test is not the same order"
            print("====================================")
            for tr_msr,te_msr,msr_name in zip(tr_type_t,te_type_t,tr_msr_name_t):
                plot_title = mangle_plot_name(cfg.modelInfo,fieldname)
                # if msr_name not in handle_dict[fieldname].keys(): handle_dict[fieldname][msr_name] = []
                if msr_name not in plot_dict.keys(): init_plot(plot_dict,plot_title,msr_name)
                add_plot_data(plot_dict,msr_name,tr_msr,fieldname,fieldvalue,'train',marker_list[marker_index[fieldname]],kmeans_search_list)
                add_plot_data(plot_dict,msr_name,te_msr,fieldname,fieldvalue,'test',marker_list[marker_index[fieldname]],kmeans_search_list)
        marker_index[fieldname] += 1

    for exp_change in unique_changes:
    # for exp_config in exp_configs:
        # update to exp_config
        # old_cfg = deepcopy(cfg)
        # update_config(cfg,exp_config)
        # fieldname,fieldvalue = what_changed(cfg,old_cfg)
        # if fieldname != expfield_filter: continue
        # reset_model_name()
        for measure_type in train_msrs.keys():
            msr_dict_type = measure_name_dict[measure_type]
            _,tr_msr_name_t = transpose_measures(train_msrs[measure_type],msr_dict_type)
            _,te_msr_name_t = transpose_measures(test_msrs[measure_type],msr_dict_type)
            for tr_msr,te_msr in zip(tr_msr_name_t,te_msr_name_t):
                assert tr_msr == te_msr,"train/test is not the same order"
            for msr_name in tr_msr_name_t:
                if msr_name not in plot_dict.keys(): init_plot(plot_dict,msr_name)
                save_name = msr_name+'_'+ exp_change + '.png'
                ylim = plot_dict[msr_name][1].get_ylim()
                xlim = plot_dict[msr_name][1].get_xlim()
                ylim = [ylim[0]-.01*ylim[1],1.01*ylim[1]]
                xlim = [xlim[0]-.05*xlim[1],1.05*xlim[1]]
                # train_handle, = plot_dict[msr_name][1].plot(None,None,'b')
                # test_handle, = plot_dict[msr_name][1].plot(None,None,'r')
                # train_test_legend = plt.legend([train_handle,test_handle],['train','test'])
                # lines = [matplotlib.lines.Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in ['b','r']]
                # train_test_legend = plt.legend(lines,['train','test'])
                # plot_dict[msr_name][1].add_artist(train_test_legend)
                train_patch = mpatches.Patch(color='blue',label='train')
                test_patch = mpatches.Patch(color='green',label='test')
                plot_dict[msr_name][1].set_ylim(ylim)
                plot_dict[msr_name][1].set_xlim(xlim)
                plot_dict[msr_name][0].subplots_adjust(right=0.7)
                plot_dict[msr_name][0].tight_layout(rect=[0,0,0.75,1])
                plot_handles = [train_patch,test_patch]
                plot_labels = ['train','test']
                handles,labels = plot_dict[msr_name][1].get_legend_handles_labels()
                plot_handles += handles
                plot_labels += labels
                plot_dict[msr_name][1].legend(handles=plot_handles,labels=plot_labels,loc='center left', bbox_to_anchor=(1, 0.5), title=exp_change)
                plot_dict[msr_name][0].savefig(save_name,bbox_inches='tight')


def analyze_expfield_results2(train_results,test_results,exp_configs,expfield_filter,kmeans_search_list):
    measure_name_dict = {'cluster':['silhouette','homogeneity_ds_labels','homogeneity_correct'],
                         'separability':['separability_ds_labels','separability_ds_correct']}
    change_field = find_experiment_changes(exp_configs)    
    fieldname_list = [field[0] for field in change_field]
    unique_changes = get_unique_strings(fieldname_list)
    marker_list = ['o','v','^','<','>','8','s','p','h','H','+','x','X','D','d']
    marker_index = dict.fromkeys(unique_changes,0)
    print(marker_index)
    plot_dict = {}
    handle_dict = {}
    for field_index,exp_config in enumerate(exp_configs):
        # update to exp_config
        old_cfg = deepcopy(cfg)
        update_config(cfg,exp_config)
        fieldname,fieldvalue = what_changed(cfg,old_cfg)
        if fieldname != expfield_filter: continue
        reset_model_name()

        # update to exp_config
        train_msrs,test_msrs = train_results[field_index],test_results[field_index]
        print(fieldname)
        print(train_msrs)
        # if fieldname not in handle_dict.keys(): handle_dict[fieldname] = {}
        for measure_type in train_msrs.keys():
            msr_dict_type = measure_name_dict[measure_type]
            print("----------------------------------")
            print(msr_dict_type)
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print(train_msrs[measure_type])
            print(test_msrs[measure_type])
            tr_type_t,tr_msr_name_t = transpose_measures(train_msrs[measure_type],msr_dict_type)
            te_type_t,te_msr_name_t = transpose_measures(test_msrs[measure_type],msr_dict_type)
            print("************************************")
            # always in the same order
            for tr_msr,te_msr in zip(tr_msr_name_t,te_msr_name_t):
                assert tr_msr == te_msr,"train/test is not the same order"
            print("====================================")
            for tr_msr,te_msr,msr_name in zip(tr_type_t,te_type_t,tr_msr_name_t):
                # if msr_name not in handle_dict[fieldname].keys(): handle_dict[fieldname][msr_name] = []
                if msr_name not in plot_dict.keys(): init_plot(plot_dict,fieldname,msr_name)
                add_plot_data(plot_dict,msr_name,tr_msr,fieldname,fieldvalue,'train',marker_list[marker_index[fieldname]],kmeans_search_list)
                add_plot_data(plot_dict,msr_name,te_msr,fieldname,fieldvalue,'test',marker_list[marker_index[fieldname]],kmeans_search_list)
        marker_index[fieldname] += 1

    for exp_change in unique_changes:
    # for exp_config in exp_configs:
        # update to exp_config
        # old_cfg = deepcopy(cfg)
        # update_config(cfg,exp_config)
        # fieldname,fieldvalue = what_changed(cfg,old_cfg)
        # if fieldname != expfield_filter: continue
        # reset_model_name()
        for measure_type in train_msrs.keys():
            msr_dict_type = measure_name_dict[measure_type]
            _,tr_msr_name_t = transpose_measures(train_msrs[measure_type],msr_dict_type)
            _,te_msr_name_t = transpose_measures(test_msrs[measure_type],msr_dict_type)
            for tr_msr,te_msr in zip(tr_msr_name_t,te_msr_name_t):
                assert tr_msr == te_msr,"train/test is not the same order"
            for msr_name in tr_msr_name_t:
                if msr_name not in plot_dict.keys(): init_plot(plot_dict,msr_name)
                save_name = msr_name+'_'+ exp_change + '.png'
                ylim = plot_dict[msr_name][1].get_ylim()
                xlim = plot_dict[msr_name][1].get_xlim()
                ylim = [ylim[0]-.01*ylim[1],1.01*ylim[1]]
                xlim = [xlim[0]-.05*xlim[1],1.05*xlim[1]]
                # train_handle, = plot_dict[msr_name][1].plot(None,None,'b')
                # test_handle, = plot_dict[msr_name][1].plot(None,None,'r')
                # train_test_legend = plt.legend([train_handle,test_handle],['train','test'])
                # lines = [matplotlib.lines.Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in ['b','r']]
                # train_test_legend = plt.legend(lines,['train','test'])
                # plot_dict[msr_name][1].add_artist(train_test_legend)
                train_patch = mpatches.Patch(color='blue',label='train')
                test_patch = mpatches.Patch(color='green',label='test')
                plot_dict[msr_name][1].set_ylim(ylim)
                plot_dict[msr_name][1].set_xlim(xlim)
                plot_dict[msr_name][0].subplots_adjust(right=0.7)
                plot_dict[msr_name][0].tight_layout(rect=[0,0,0.75,1])
                plot_handles = [train_patch,test_patch]
                plot_labels = ['train','test']
                handles,labels = plot_dict[msr_name][1].get_legend_handles_labels()
                plot_handles += handles
                plot_labels += labels
                plot_dict[msr_name][1].legend(handles=plot_handles,labels=plot_labels,loc='center left', bbox_to_anchor=(1, 0.5))
                plot_dict[msr_name][0].savefig(save_name)



if __name__ == "__main__":
    print("This file is provided for mangling the results from:")
    print("-> the optimal_k process from fresh.py")
    print("-> the separability of the original model on each dataset")

    # get experiment list
    experiment_configs = generate_experiments_across_iterations()

    # print experiment info
    # print(cfg)
    # pp(experiment_configs)

    # start cache
    expCache = Cache("experiment_results.pkl",cfg,'results')

    # run experiments
    train_results = []
    test_results = []
    for experiment in experiment_configs:
        update_config(cfg,experiment)
        print(cfg)
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


        
