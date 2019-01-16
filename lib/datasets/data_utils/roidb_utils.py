from utils.base import *
import numpy as np
import numpy.random as npr
import cv2

#
# functions for creating the roidb
#

def subsample_roidb(gt_roidb,image_index,subsample_size):
    # --> SUBSAMPLE THE DATA <--
    if subsample_size != -1:
        original_size = npr.permutation(len(gt_roidb))
        shuffled_indices = npr.permutation(original_size)[:subsample_size]
        gt_roidb = [gt_roidb[index] for index in shuffled_indices]
        filtered_image_index = [image_index[index] for index in shuffled_indices]
        return gt_roidb,filtered_image_index
    else:
        return gt_roidb,image_index        

def filterSampleWithEmptyAnnotations(gt_roidb,image_index):
    toRemove = []
    print("[roidb_utils.py: filterSampleWithEmptyAnnotations]")
    for idx,sample in enumerate(gt_roidb):
        if len(sample['gt_classes']) == 0: toRemove.append(idx)
    gt_roidb,filtered_image_index = removeListFromGtRoidb(gt_roidb,image_index,toRemove)
    return gt_roidb,filtered_image_index

def filterImagesByClass(gt_roidb,image_index,class_filter):
    print("[roidb_utils.py: filterImagesByClass]")
    if len(class_filter.new_names) == 0:
        print("no classes to filter")
        return gt_roidb,image_index
    toRemove = []
    for idx,sample in enumerate(gt_roidb):
        keepBool = filterSampleByClass(sample,class_filter)
        if keepBool is False:
            toRemove.append(idx)
    gt_roidb,filtered_image_index = removeListFromGtRoidb(gt_roidb,image_index,toRemove)
    return gt_roidb,filtered_image_index

def filterSampleByClass(sample,class_filter):
    # if no gt_classes match, return None
    # gt_class_names = [class_filter.original_names[cls] for cls in sample['gt_classes']]
    # class_inclusion_list = class_filter.new_names
    if check_list_equal_any(sample['gt_classes'],-1) is False:
        return True
    # if all gt_classes match, return original input
    if check_list_equal_all(sample['gt_classes'],-1) is True:
        return False
    # filter out all other samples; we should never have zero remaining because of first "if" check
    toRemove = []
    for gt_obj_index,gt_class_index in enumerate(sample['gt_classes']):
        print(gt_class_index)
        if gt_class_index != -1:
            continue
        else:
            toRemove.append(gt_obj_index)
    for fieldname,valueItem in sample.items():
        if hasattr(valueItem, '__len__'): # if there is a list 
            for idx in sorted(toRemove,reverse=True):
                del sample[fieldname][idx]
    return True

def removeListFromGtRoidb(gt_roidb,image_index,toRemove):
    numFiltered = len(toRemove)
    filtered_image_index = list(image_index)
    for idx in sorted(toRemove,reverse=True):
        del gt_roidb[idx]
        del filtered_image_index[idx]
    print("filtered {} samples".format(numFiltered))
    return gt_roidb,filtered_image_index

#
# flattening the roidb
#

def flattenRoidbDict(roidbDict,dataset_names_ordered,numSamples=None):
    roidbFlattened = []
    for key,roidb in roidbDict.items():
        print("{}: {} images".format(key,len(roidb)))
        toExtend = roidb
        if numSamples is not None:
            index = dataset_names_ordered.index(key)
            sizeToKeep = numSamples[index]
            if sizeToKeep is not None:
                print("[flattenRoidbdict] shortened roidb from {} to {}".format(len(roidb),sizeToKeep))
            else:
                sizeToKeep = len(roidb)
            toExtend = roidb[:sizeToKeep]
        roidbFlattened.extend(toExtend)
    return roidbFlattened

#
# predefined values to grab values from roidb
#

def roidbSampleImage(sample,annoIndex):
    # load the image
    img = cv2.imread(sample['image'])
    if sample['flipped']:
        img = img[:, ::-1, :]
    return img,sample['set']

def roidbSampleImageHOG(sample,annoIndex):
    # load the image
    if sample['flipped']:
        print("Error: can't use flipped images")
        sys.exit()
    return sample["hog_image"],sample['set']

def roidbSampleImageAndBox(sample,annoIndex):
    # load the image
    img = cv2.imread(sample['image'])
    if sample['flipped']:
        img = img[:, ::-1, :]
    return [img,sample['boxes'][annoIndex]],sample['set']

def roidbSampleHOG(sample,annoIndex):
    # load the hog
    return sample['hog'][annoIndex],sample['set']

def roidbSampleBox(sample,annoIndex):
    # load the box
    return sample['boxes'][annoIndex],sample['set']

#
# common useful function
#

def compute_size_along_roidb(roidb):
    _roidbSize = []
    if roidb is None:
        raise ValueError("roidb must be loaded before 'compute_size_along_roidb' can be run")
    _roidbSize.append(len(roidb[0]['boxes']))
    for image in roidb[1:]:
        newSize = _roidbSize[-1] + len(image['boxes'])
        _roidbSize.append(newSize)
    return _roidbSize

def addRoidbField(roidb,fieldName,transformFunction):
    firstRoidb = getFirstElementNotNone(roidb)
    if fieldName in firstRoidb.keys():
        print("WARNING: field name [{}] already exists.".format(fieldName))

    totalRoidbs = len(roidb)
    for idx,sample in enumerate(roidb):

        # progress = int(float(idx) / float(totalRoidbs) * 100)
        # print("-"*progress)
        # print("")

        if fieldName in sample.keys() and sample[fieldName] is not None:
            continue
            # sample[fieldName].append(transformFunction(sample))
        sample[fieldName] = transformFunction(sample)
        

#
# start: class filter functions
#

def mangleClassInclusionList(class_inclusion_list,gt_roidb):
    if len(class_inclusion_list) == 0:
        return class_inclusion_list
    if type(class_inclusion_list[0]) == type(gt_roidb[0]['gt_classes'][0]):
        return class_inclusion_list
    elif type(class_inclusion_list[0]) is str and ( type(gt_roidb[0]['gt_classes'][0]) is np.uint8 or type(gt_roidb[0]['gt_classes'][0]) is int):
        return [int(cls) for cls in class_inclusion_list]
    elif type(class_inclusion_list[0]) is int and type(gt_roidb[0]['gt_classes'][0]) is str:
        return [str(cls) for cls in class_inclusion_list]
    else:
        print("ERROR. How do we match the 'class_inclusion_list' type with the 'gt_roidb['gt_classes']' type?")
        exit()

def createClassConversionDict(class_inclusion_list,original_classes):
    print(class_inclusion_list)
    print(original_classes)
    conversion_dict = {}
    for cls in class_inclusion_list:
        conversion_dict[original_classes.index(cls)] = cls
    return conversion_dict

#
# end:
#

#
# misc roidb functions
#

def computeRoidbDictLens(roidbTrDict,roidbTeDict):
    lenTr = 0
    for roidb in roidbTrDict.values():
        lenTr += len(roidb)
    lenTe = 0
    for roidb in roidbTeDict.values():
        lenTe += len(roidb)

    return lenTr,lenTe

def combine_roidb(self,roidbs):
    # assumes ordering of roidbs
    roidb = []
    for r in roidbs:
        # skips "None"; shouldn't impact the outcome
        if r is None: continue
        roidb.extend(r)
        print_each_size(roidb)
    return roidb

def combineOnlyNewRoidbs(roidbs,pc):
    # assumes ordering of roidbs
    newRoidb = []
    print(pc)
    for idx,roidb in enumerate(roidbs):
        if roidb is None: continue
        newRoidb.extend(roidb[pc[idx]:])
    return newRoidb

#
# printing functions
#

def print_each_size(roidb):
    print("="*100)
    sizes = [0 for _ in range(8)]
    for elem in roidb:
        sizes[elem['set']] += len(elem['boxes'])
    print(sizes)

def printRoidbImageIds(roidb):
    for sample in roidb:
        print(sample['image'])

def printRoidbDictImageNamesToTextFile(roidbDict,postfix):
    fid = open("output_{}.txt".format(postfix),"w+")
    for key in roidbDict.keys():
        for idx,sample in enumerate(roidbDict[key]):
            fid.write(sample['image']+"\n")
    fid.close()
    
def printRoidbImageNamesToTextFile(roidb,postfix):
    fid = open("output_{}.txt".format(postfix),"w+")
    for sample in roidb:
        print(sample['image'])
        fid.write(sample['image']+"\n")
    fid.close()
