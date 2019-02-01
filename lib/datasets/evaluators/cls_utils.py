import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pickle,sys,os
from datasets import ds_utils
from core.config import cfg,cfgData,load_tp_fn_record_path
import os.path as osp
from easydict import EasyDict as edict

from utils.base import *
from utils.misc import computeEntropyOfNumpyArray

from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from utils.plot_utils import *


cfg._DEBUG.datasets.evaluators.cls_utils = False


def createAugmentationString(aug_index_str,augmentation_info):
    aug_string = ''
    for type_index,type_str in enumerate(augmentation_info.config_order):
        for param_index in augmentation_info[type_str].config_order:
            aug_info = str(augmentation_info.configs[int(aug_index_str)][type_index][param_index])
            if int(aug_index_str) is -1 or aug_info is None or aug_info is False:
                aug_info = '-'
            aug_string += ',' + aug_info
    return aug_string

def createAugmentationHeader(augmentation_info):
    aug_string = ''
    for type_index,type_str in enumerate(augmentation_info.config_order):
        for param_index in augmentation_info[type_str].config_order:
            aug_string += ',{}({})'.format(type_str,param_index)
    for data_string in augmentation_info.data_order:
        aug_string += ','+data_string
    return aug_string

def createCsvHeader(csv_information):
    # field_names_in_order = csv_information.field_names_in_order
    header_str = 'class_name'
    # header_str += ','.join(field_names_in_order)
    aug_header = createAugmentationHeader(csv_information.augmentation_info)
    header_str += aug_header
    header_str += '\n'
    return header_str
    
def write_evaluation_to_csv(evaluation,csv_information):
    #field_names_in_order = csv_information.field_names_in_order
    write_str = ''
    header_string = createCsvHeader(csv_information)
    write_str += header_string
    # for index in evaluation.shape[0]:
    #     field_values = evaluation[index,:]
    #     for value in field_values:
    #         write_str = ','+value
    #     write_str = '\n'
    for class_name in evaluation.classes.keys():
        for aug_index_str in evaluation.classes[class_name].augmentations.keys():
            # class_info = csv_translations.classes[class_name] hypothetical use of this
            aug_string = createAugmentationString(aug_index_str,csv_information.augmentation_info)
            data = evaluation.classes[class_name].augmentations[aug_index_str].data
            print(data)
            write_str += class_name
            write_str += aug_string
            for data_type in csv_information.augmentation_info.data_order:
                write_str += ',' + str(data[data_type])
            write_str += '\n'

    print(write_str)
    # with open("results_filename.txt") as f:
    #     fid.write(write_str


def print_evaluations(evaluation):
    if 'data' in evaluation.keys():
        for name,value in evaluation.data.items():
            print(name,value)
    for name,value in evaluation.items():
        if name == 'data':
            continue
        print(name)
        print_evaluations(value)
    return


def metrics_by_class(info_by_field,agg_evaluation,classes,augmentations):

    confusion_matrix = sk_confusion_matrix(info_by_field[:,1],info_by_field[:,2])
    for index in range(len(classes)):
        name = classes[index]
        indices = np.where(info_by_field[:,2] == index)[0]
        num_samples = len(indices)
        class_info = info_by_field[indices,:]
        if len(class_info) == 0:
            continue
        class_metrics = compute_evaluation_metrics(confusion_matrix,index)
        class_metrics['accuracy'] = np.mean(class_info[:,3].astype(np.float32))
        class_metrics['entropy'] = computeEntropyOfNumpyArray(class_info[:,0].astype(np.float32))
        class_metrics['number_of_samples'] = num_samples
        accuracy = class_metrics['accuracy']
        print("{}: acc({}) #samples({})".format(name,accuracy,num_samples))

        # add results to aggregation
        agg_evaluation[name] = edict()
        agg_evaluation[name].data = edict(class_metrics)
        agg_evaluation[name].augmentations = edict()
        metrics_by_augmentation(class_info,agg_evaluation[name].augmentations,augmentations)
        


def augmenationConfigToListValues(augmentation):
    a = int(augmentation[0]['flip'])
    b = int(augmentation[1]['step'])
    c = int(augmentation[1]['direction'])
    d = float(augmentation[2]['angle'])
    e = float(augmentation[3]['step'])
    list_values = [a,b,c,d,e]
    return list_values

def getAugmentationIndices(evaluations,augmentation):
    list_values = augmenationConfigToListValues(augmentation)
    indices = np.where(np.all(evaluations[:,4:] == list_values,axis=1))[0]
    return indices

def metrics_by_augmentation(info_by_field,agg_evaluation,augmentations):
    num_augmentations = len(augmentations)
    for aug_index in range(-1,num_augmentations):
        info_str = str(aug_index)
        indices = getAugmentationIndices(info_by_field,augmentations[aug_index])
        num_samples_of_aug = len(indices)
        print("num_samples_of_aug: {}".format(num_samples_of_aug))
        if num_samples_of_aug > 0:
            filtered_samples = info_by_field[indices,:]
            accuracy = np.mean(filtered_samples[:,3])
            entropy = computeEntropyOfNumpyArray(filtered_samples[:,0])
        else:
            accuracy = '-'
            entropy = '-'
        augmentation_metrics = edict()
        augmentation_metrics.data = edict()
        augmentation_metrics.data.accuracy = accuracy
        augmentation_metrics.data.entropy = entropy
        agg_evaluation[info_str] = augmentation_metrics

def compute_evaluation_metrics(confusion_matrix,index):
    # extract some confusion matrix info 
    number_of_samples = np.sum(confusion_matrix)
    true_positives = confusion_matrix[index,index]
    false_positives = np.sum(confusion_matrix[:,index]) - true_positives
    false_negatives = np.sum(confusion_matrix[index,:]) - true_positives
    true_negatives = number_of_samples - true_positives - false_positives -false_negatives

    # compute some derived quantities
    # accuracy = ( true_positives + true_negatives ) / ( 1. * number_of_samples )
    f1_score = ( 2. * true_positives ) / ( 2. * true_positives + false_positives + false_negatives )
    sensitivity = true_positives / (1. * true_positives + false_negatives )
    specificity = true_negatives / (1. * true_negatives + false_positives )
    precision_positive = true_positives / (1. * true_positives + false_positives )
    precision_negative = true_negatives / (1. * true_negatives + false_negatives )
    false_negative_rate = false_negatives / (1. * false_negatives + true_positives )
    false_positive_rate = false_positives / (1. * false_positives + true_negatives )

    metrics = {'f1_score':f1_score,
               'sensitivity':sensitivity,
               'true_positives':true_positives,
               'false_positives':false_positives,
               'false_negatives':false_negatives,
               'true_negatives':true_negatives,
               'specificity':specificity,
               'precision_positive':precision_positive,
               'precision_negative':precision_negative,
               'false_negative_rate':false_negative_rate,
               'false_positive_rate':false_positive_rate
    }

    return metrics



def filter_evaluations(evaluations,field_filters):
    eq_check_list = []
    for field_index,field_value in enumerate(field_filters):
        if field_value is not None:
            eq_check = np.isclose(evaluations[:,field_index],field_value)
            eq_check_list.append(eq_check)
    # "AND" all the masks
    eq_final = np.ones(eq_check_list[0].shape,dtype=np.bool)
    for eq_check in eq_check_list:
        eq_final = np.logical_and(eq_final,eq_check)
    # get indicies for all true
    indices = np.where(eq_final)[0]
    if len(indices) == 0:
        return [],[]
    else:
        filtered_evaluations = evaluations[indices,:]
        return filtered_evaluations,indices

def plot_rotation_by_class(evaluation,classes,angles,net_name):

    figure,axis = init_plot("accuracy under rotation by class",'degrees','accuracy')
    eval_filter = [None for _ in range(evaluation.shape[1])]
    for class_index,class_name in enumerate(classes):
        angle_accuracy_list = []
        angle_list = []
        for angle in angles:
            eval_filter[2] = class_index
            eval_filter[7] = angle
            filtered_evaluations,_ = filter_evaluations(evaluation,eval_filter)
            if len(filtered_evaluations) == 0:
                print("no samples for (class_index,angle) = ({},{})".format(class_index,angle))
                continue
            accuracy = np.mean(filtered_evaluations[:,3])
            angle_list.append(angle)
            angle_accuracy_list.append(accuracy)
            print(angle,accuracy,len(filtered_evaluations))
        axis.plot(angle_list,angle_accuracy_list,label=class_name)

    # get overall accuracy
    overall_accuracy_by_angle = []
    eval_filter = [None for _ in range(evaluation.shape[1])]
    for angle in angles:
        eval_filter[7] = angle
        filtered_evaluations,_ = filter_evaluations(evaluation,eval_filter)
        if len(filtered_evaluations) == 0:
            print("no samples for (class_index,angle) = ({},{})".format(class_index,angle))
            continue
        accuracy = np.mean(filtered_evaluations[:,3])
        overall_accuracy_by_angle.append(accuracy)
    axis.plot(angles,overall_accuracy_by_angle,label="overall")

    filename = 'rotation_{}.png'.format(net_name)
    save_plot(filename,figure,axis)

def plot_rotation_with_layers_angle(evaluations,aggActivations,angles,classes):
    model_angles = aggActivations.agg_obj['warp_angle'] * 90
    net_name = aggActivations.save_cache.test_cache_config.modelInfo.name
    eval_filter = [None for _ in range(evaluations.shape[1])]
    num_classes = len(classes)
    for class_index,class_name in enumerate(classes):
        figure,axis = init_plot("model angle v.s. angle",'data angle','model angle')
        ave_model_angle_list = []
        std_model_angle_list = []
        lower_ribbon = []
        upper_ribbon = []
        for angle in angles:
            eval_filter[2] = class_index
            eval_filter[7] = angle
            filtered_evaluations,indices = filter_evaluations(evaluations,eval_filter)
            filtered_model_angles = model_angles[indices,:]
            ave = np.mean(filtered_model_angles)
            std = np.std(filtered_model_angles)/np.sqrt(len(filtered_model_angles))
            ave_model_angle_list.append(ave)
            std_model_angle_list.append(std)
            lower_ribbon.append(ave-1.6*std)
            upper_ribbon.append(ave+1.6*std)
        color = [class_index / (1.*num_classes),0.25,0.75]
        axis.fill_between(angles,lower_ribbon,upper_ribbon,label=class_name,color=color,alpha=0.4)
        filename = 'model_angle_{}_{}.png'.format(class_name,net_name)
        save_plot(filename,figure,axis)

#
# functions to create derived datasets
#

