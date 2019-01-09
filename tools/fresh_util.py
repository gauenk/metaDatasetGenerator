from fresh_config import cfg,cfgForCaching,checkConfigEquality,update_one_current_config_field

from datasets.factory import get_repo_imdb
from datasets.ds_utils import loadRecord,loadActivityVectors
from utils.base import readPickle,writePickle

import numpy as np
import uuid


class FreshData():

    def __init__(self,samples,records,data_class_assignment,classes):
        self.samples = samples
        self.correct = records
        self.ds_labels = data_class_assignment
        self.classes = classes
        self.exp_samples = None
        self.exp_labels = None

class Cache():

    """
    -> filename: the location of the cache
    -> stores within the filename by a 'uuid'
    -> search by comparing the config
    """

    def __init__(self,filename,config,fieldname=None):
        self.filename = filename
        self.fieldname = fieldname
        self.config = config
        self.is_valid = False

    def load(self):
        if cfg.cache is False: return None
        cache = readPickle(self.filename)
        if cache is None or len(cache) == 0: return None
        self.is_valid = True
        for uuID,expData in cache.items():
            self.is_valid = True
            isEqual = checkConfigEquality(expData['config'],self.config)
            #isEqual = checkConfigEquality(self.config,expData['config'])
            if isEqual:
                if self.fieldname is not None:
                    if self.fieldname in expData['data'].keys(): return expData['data'][self.fieldname]
                    else:
                        self.is_valid = False
                        return None
                else: return expData['data']
        self.is_valid = False
        return None

    def save(self,payload,saveField=""):
        if cfg.cache is False:
            print("Did not save. Caching is turned off.")
            return None
        if saveField == "": saveField = self.fieldname
        cache = readPickle(self.filename)
        if cache is None: cache = {}
        uuID = str(uuid.uuid4())
        blob = {'config':self.config}
        blob['data'] = {}
        if saveField is not None: blob['data'][saveField] = payload
        else: blob['data'] = payload
        cache[uuID] = blob
        writePickle(self.filename,cache)

    def view(self):
       cache = readPickle(self.filename)
       print(cache.keys())
       for key,value in cache.items():
           print(key,value['config'].expName,checkConfigEquality(value['config'],cfg))

def transformData(data):
    if cfg.data.transformations.apply_relu:
        data = data[np.where(data < 0)[0]]
    if cfg.data.transformations.normalize:
        if data.max() == 0:
            return data
        norm_data = (data + data.min()) / data.max()
        return norm_data
    elif cfg.data.transformations.to_bool:
        return (data >= 0)
    else:
        return data

def compute_accuracy(truth,guess):
    return np.mean(truth == guess)

def getGroundtruthClassLabels_extra(imdb_name,numberOfSamples=-1):
    print("WARNING: dont use! only keeping because i am not ready to delete.")
    imdb = get_repo_imdb(imdb_name)
    if numberOfSamples == -1: numberOfSamples = len(imdb.image_index)
    imageIndexIDs = imdb.image_index[:numberOfSamples]
    imageIndexIDs_sortedIndex = sorted(range(len(imageIndexIDs)), key=lambda k: imageIndexIDs[k])
    imageIndexIDs_sortedIndex = imageIndexIDs_sortedIndex
    roidb = imdb.roidb
    gt_labels = np.zeros(len(imageIndexIDs_sortedIndex),dtype=np.uint8)
    for gt_index,roidb_index in enumerate(imageIndexIDs_sortedIndex):
        gt_labels[gt_index] = roidb[roidb_index]['gt_classes'][0]
    return gt_labels,imdb.classes,sorted(imageIndexIDs)

def getGroundtruthClassLabels(imdb_name,numberOfSamples=-1):
    imdb = get_repo_imdb(imdb_name)
    roidb = imdb.roidb
    if numberOfSamples == -1: numberOfSamples = len(imdb.image_index)
    imageIndexIDs = sorted(imdb.image_index) # always sort the ENTIRE image_index
    gt_labels = np.zeros(len(imageIndexIDs),dtype=np.uint8)
    for gt_index,imageID in enumerate(imageIndexIDs[:numberOfSamples]):
        roidb_index = imdb.image_index.index(imageID)
        gt_labels[gt_index] = roidb[roidb_index]['gt_classes'][0]
    return gt_labels,imdb.classes,sorted(imageIndexIDs)

def aggregateActivations(av,imageIndexList,numberOfSamples=-1,verbose=False):
    # init combination dictinaries
    allCombo = {}
    comboInfo = cfg.comboInfo
    for comboID in comboInfo:
        allCombo[comboID] = []

    # aggregate activation information by combination settings
    if numberOfSamples == -1: numberOfSamples = len(imageIndexList)
    if type(av[cfg.layerList[0]]) is not dict:
        imageIndexList = range(numberOfSamples)
        for comboID in comboInfo:
            layerNames = comboID.split("-")
            allComboList = []
            # single samples (e.g. one row; we are building the features)
            for layerName in layerNames:
                layer_activations = av[layerName][:numberOfSamples,:].reshape(numberOfSamples,-1)
                allComboList.append(layer_activations)
            # add the single sample to the data list
            allComboList = np.hstack(allComboList)
            allCombo[comboID] = allComboList
    else:
        for imageIndexID in imageIndexList[:numberOfSamples]:
            for comboID in comboInfo:
                layerNames = comboID.split("-")
                allComboList = []
                # single samples (e.g. one row; we are building the features)
                for layerName in layerNames:
                    activations = av[layerName][imageIndexID].ravel()
                    allComboList.extend(activations)
                # add the single sample to the data list
                allComboListTrans = transformData(allComboList)
                print(allComboListTrans)
                allCombo[comboID].append(allComboList)

    # "numpify" all the lists per combo per class
    for comboID in comboInfo:
        allCombo[comboID] = np.array(allCombo[comboID])
        if cfg.verbose or verbose:
            print("all [#samples x #ftrs]",comboID,allCombo[comboID].shape)
        
    return allCombo
    
def load_and_reorder_records(imdb_name,modelInfo,imageIndexIDs):
    records_dict = loadRecord(imdb_name,modelInfo,cfg.load_record_pickle)
    if cfg.load_record_pickle:
        records = np.zeros((len(imageIndexIDs)),dtype=np.uint8)
        for record_index,imageIndex in enumerate(imageIndexIDs):
            records[record_index] = records_dict[imageIndex][0]
    else: records = records_dict.astype(np.uint8)
    return records

def load_data(imdb_name,layerList,modelInfo):
    # cache_name = "data_debug_cache_" + imdb_name + ".pkl"
    # expDataCache = Cache(cache_name,cfgForCaching)
    # dataset = expDataCache.load()
    # if expDataCache.is_valid: return dataset

    class_labels,classes,imageIndexIDs = getGroundtruthClassLabels(imdb_name,
                                                                   cfg.data.numberOfSamples)
    records = load_and_reorder_records(imdb_name,modelInfo,imageIndexIDs)
    av = loadActivityVectors(imdb_name,layerList,modelInfo.name,load_pickle=cfg.load_av_pickle,load_bool=cfg.load_bool_activity_vectors)
    av_combos = aggregateActivations(av,imageIndexIDs,cfg.data.numberOfSamples) 
    # ^one line above^ takes 90% of startup time

    dataset = FreshData(None,records,class_labels,classes)
    dataset.samples = {}
    for comboID,combo in av_combos.items():
        a,b,c =  combo.shape[0],len(records),len(class_labels)
        assert a == b, "not the same length [records]: {} vs {}".format(a,b)
        assert a == c, "not the same length [class_labels]: {} vs {}".format(a,c)
        dataset.samples[comboID] = combo

    # expDataCache.save(dataset)
    return dataset

def get_unique_strings(alist):
    output = []
    for x in alist:
        if x not in output:
            output.append(x)
    return output

def list_of_floats_to_string(list_of_floats):
    list_str = ', '.join(map('{:.03f}'.format, list_of_floats))
    return list_str

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


