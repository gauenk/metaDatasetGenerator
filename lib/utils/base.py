import os,cPickle,sys
import os.path as osp
import numpy as np
import cv2

def print_warning(scriptname,input_string):
    print("\n\n\n\n\n\n\n\n\n")
    print("-"*100)
    print("WARNING:")
    print(scriptname)
    print(input_string)
    print("-"*100)
    print("\n\n\n\n\n\n\n\n\n")

def readPickle(fn):
    print("[readPickle]: {} *pickle pickle*".format(fn))
    if not osp.exists(fn):
        print("[readPickle]: no existing file named: {}".format(fn))
        return None
    tmp = None
    with open(fn,'rb') as f:
        tmp = cPickle.load(f)
    if tmp is None: print("ERROR reading picklefile: {}".format(fn))
    return tmp

def writePickle(fn,data):
    print("[writePickle]: {} *sluurrrp*".format(fn))
    with open(fn,'wb') as f:
        cPickle.dump(data,f,cPickle.HIGHEST_PROTOCOL)

def readNdarray(fn):
    print("[readNdarray]: {} *zoom zoom*".format(fn))
    if not osp.exists(fn):
        print("[readNdarray]: no existing file named: {}".format(fn))
        return None
    ndarray = np.load(fn)
    return ndarray

def writeNdarray(fn,data):
    print("[writeNdarray]: {} *cha-chunk*".format(fn))
    np.save(fn,data)

def scaleImageScalar(im_orig,target_size):
    x_size,y_size = im_orig.shape[0:2]
    if x_size == 0 or y_size == 0:
        print("WARNING: image's size 0 for an axis")
        return im_orig
    im = cv2.resize(im_orig, (target_size,target_size), interpolation=cv2.INTER_CUBIC)
    return im

def scaleImageTuple(im_orig,target_size):
    x_size,y_size = im_orig.shape[0:2]
    if x_size == 0 or y_size == 0:
        print("WARNING: image's size 0 for an axis")
        return im_orig
    im = cv2.resize(im_orig, (target_size[0],target_size[1]), interpolation=cv2.INTER_CUBIC)
    return im

def scaleImage(im_orig,target_size):
    if hasattr(target_size, '__len__'):
        return scaleImageTuple(im_orig,target_size)
    else:
        return scaleImageScalar(im_orig,target_size)
        
def check_list_equal_any(list_a,input_b):
    if type(input_b) is list:
        return check_list_and_list_equal_any(list_a,input_b)
    else:
        return check_list_and_value_equal_any(list_a,input_b)
    # else:
    #     print(type(list_a[0]), type(input_b))
    #     raise TypeError("unknown how to handle type of input_b")

def check_list_and_value_equal_any(list_a,elem_b):
    for elem_a in list_a:
        if elem_a == elem_b:
            return True
    return False

def check_list_and_list_equal_any(list_a,list_b):
    if len(list_a) == 0 or len(list_b) == 0:
        return False
    for item_a in list_a:
        for item_b in list_b:
            if item_a == item_b:
                return True
    return False
    
def check_list_equal_all(list_a,input_b):
    if type(input_b) is list:
        return check_list_and_list_equal_any(list_a,input_b)
    else:
        return check_list_and_value_equal_any(list_a,input_b)
    # elif type(list_a[0]) is type(input_b):
    #     check_list_and_value_equal_any(list_a,input_b)
    # else:
    #     raise TypeError("unknown how to handle type of input_b")

def check_list_and_value_equal_all(list_a,elem_b):
    for elem_a in list_a:
        if elem_a != elem_b:
            return False
    return True

def check_list_and_list_equal_all(list_a,list_b):
    if len(list_a) == 0 or len(list_b) == 0:
        return False
    for item_a in list_a:
        for item_b in list_b:
            if item_a != item_b:
                return False
    return True

def getFirstElementNotNone(pythonList):
    return next(item for item in pythonList if item is not None)

def transformNumpyData(data,transformations):

    if transformations['apply_relu']:
        data = data[np.where(data < 0)[0]]

    if transformations['normalize']:
        if data.max() == 0:
            return data
        norm_data = (data + data.min()) / data.max()
        return norm_data
    elif transformations['to_bool']:
        return (data >= 0)
    else:
        return data

def listToString(list_a):
    rlist = [str(item) for item in list_a]
    return rlist

def get_unique_strings(alist):
    output = []
    for x in alist:
        if x not in output:
            output.append(x)
    return output

def list_of_floats_to_string(list_of_floats):
    list_str = ', '.join(map('{:.03f}'.format, list_of_floats))
    return list_str

def compute_accuracy(truth,guess):
    return np.mean(truth == guess)

def check_number_of_used_augmentation_fields(evaluation,fieldnames):
    augmentations_in_use = {}
    for index,fieldname in enumerate(fieldnames):
        if 'dataset_augmentation' in fieldname:
            inuse = len(np.unique(evaluation[:,index])) > 1
            augmentations_in_use[fieldname] = inuse
    number_augs_in_use = sum([inuse for aug_field,inuse in augmentations_in_use.items()])
    print("there are {} augmentation(s) in use".format(number_augs_in_use))
    print("the augmentation fields in use are:")
    print(augmentations_in_use)
    return number_augs_in_use

