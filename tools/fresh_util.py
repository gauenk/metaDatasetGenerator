from fresh_config import cfg,cfgForCaching,checkEdictEquality,update_one_current_config_field

from datasets.factory import get_repo_imdb
from datasets.data_utils.records_utils import loadRecord
from datasets.data_utils.activation_value_utils import loadActivationValues,createActivationValuesComboDict
from utils.base import readPickle,writePickle
from cache.one_level_cache import Cache

import numpy as np
import uuid


class DataWrapper():
    def __init__(self,activation_combos,records,data_labels,classes):
        self.samples = {}
        self.correct = records
        self.ds_labels = data_labels
        self.classes = classes
        self.exp_samples = None
        self.exp_labels = None
        dataset.samples = {}
        for comboID,combo in activation_combos.items():
            a,b,c =  combo.shape[0],len(records),len(data_labels)
            assert a == b, "not the same length [records]: {} vs {}".format(a,b)
            assert a == c, "not the same length [data_labels]: {} vs {}".format(a,c)
            dataset.samples[comboID] = combo

def load_data(imdb_name,layerList,modelInfo):
    imdb = get_repo_imdb(imdb_name)
    classes = imdb.classes
    roidb_labels = imdb.get_roidb_labels()
    record = loadRecord(imdb_name,modelInfo)
    activation_values = loadActivationValues(imdb_name,layerList,modelInfo.name,load_pickle=cfg.load_av_pickle,load_bool=cfg.load_bool_activity_vectors)
    activation_combos = createActivationValuesComboDict(activation_values,cfg.comboInfo,cfg.data.numberOfSamples)
    # ^one line above^ takes 90% of startup time
    dataset = DataWrapper(activation_combos,records,class_labels,classes)
    # expDataCache.save(dataset)
    return dataset

def print_result_list(list_of_lists,list_with_meaningful_values):
    for index,m_index in enumerate(list_with_meaningful_values):
        list_str = list_of_floats_to_string(list_of_lists[index])
        print("@ {}: [{}]".format(m_index,list_str))

def print_measure_results(measure_list,kmeans_search_list):
    """
    measure_list:
    -> comboID
       -> cluster statistics
    -> separation dataset classes
    -> separation correct/incorrect
    """
    print(measure_list)

def aggregate_results(measure_list,number_of_k):
    agg_measure_score = [0 for _ in range(number_of_k)]
    for measure_type in measure_list.keys():
        type_list = measure_list[measure_type]
        if measure_type == 'cluster':
            for k_index in range(number_of_k):
                for comboID_results in type_list[k_index].values():
                    agg_measure_score[k_index] += sum(comboID_results)
        elif measure_type == 'separability':
            for k_index in range(number_of_k):
                agg_measure_score[k_index] += sum(type_list[k_index])
        else:
            print("unknown measure type: {}".format(measure_type))
        
    # for k_index in range(number_of_k):

    return agg_measure_score

def transpose_measures(measure_list,measure_name_list):
    if type(measure_list[0]) is list: return list_transpose(measure_list),measure_name_list
    if type(measure_list[0]) is dict: return combo_list_transpose(measure_list,measure_name_list)
    

def combo_list_transpose(measure_list,measure_name_list):
    cluster_combo = {}
    measure_name_list_all = []
    number_of_measures = len(measure_name_list)
    number_of_comboids = len(measure_list[0])
    number_of_rows = number_of_measures * number_of_comboids
    number_of_ks = len(measure_list)
    combos = [ [ 0 for _ in range(number_of_ks) ] for _ in range(number_of_rows) ]
    measure_name_list_all = [ None for _ in range(number_of_rows) ]
    for k_index in range(len(measure_list)):
        for comboIndex,comboID in enumerate(measure_list[k_index].keys()):
            for measure_index in range(number_of_measures):
                row_index = measure_index + comboIndex * number_of_measures
                row_name = measure_name_list[measure_index] + "+" + comboID
                combos[row_index][k_index] = measure_list[k_index][comboID][measure_index]
                measure_name_list_all[row_index] = row_name
    return combos, measure_name_list_all
        

def list_transpose(list_of_lists):
    # reorder major and minor axis of "2d" list... assuming each column length is equal
    new_list = [ [0 for _ in range(len(list_of_lists)) ] for idx in range(len(list_of_lists[0])) ]
    for idx in range(len(list_of_lists)):
        for jdx in range(len(list_of_lists[idx])):
            new_list[jdx][idx] = list_of_lists[idx][jdx]
    return new_list


def assert_equal_lists(alist,blist):
    for a,b in zip(alist,blist): assert a == b,"lists not the same"

def find_experiment_list_changes(exp_configs):
    change_field = []
    for exp_config in exp_configs:
        fieldname,fieldvalue = update_one_current_config_field(exp_config)
        change_field.append([fieldname,fieldvalue])
    return change_field

def get_unique_experiment_field_change(exp_configs):
    change_field = find_experiment_list_changes(exp_configs)
    fieldname_list = [field[0] for field in change_field]
    unique_changes = get_unique_strings(fieldname_list)
    return unique_changes


