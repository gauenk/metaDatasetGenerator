#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2,sys
import itertools
import os
import glob
import time
import math
from core.config import cfg
from random import shuffle
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from datasets.ds_utils import cropImageToAnnoRegion,addRoidbField,clean_box,scaleRawImage
from sklearn.calibration import CalibratedClassifierCV as cccv
import warnings
warnings.filterwarnings('ignore')

# cfg.DEBUG = True

def bboxHOGfromRoidbSample(sample,orient=9, pix_per_cell=8,
                       cell_per_block=2, hog_channel=0):
    features = []
    if 'image' not in sample.keys():
        # print(sample)
        print(sample.keys())
        print("WARINING [bboxHOGfromRoidbSample]: the\
        image field is not available for the above sample")
        return None
    img = cv2.imread(sample['image'])
    for box in sample['boxes']:
        box = clean_box(box,sample['width'],sample['height'])
        cimg = cropImageToAnnoRegion(img,box)
        feature_image = np.copy(cimg)      
        try:
            features.append(HOGFromImage(feature_image))
        except Exception as e:
            print(e)
            print('hog failed @ path {}'.format(sample['image']))
            return None
    return features

def imageHOGfromRoidbSample(sample,orient=9, pix_per_cell=8,
                       cell_per_block=2, hog_channel=0):
    if 'image' not in sample.keys():
        #print(sample)
        print(sample.keys())
        print("WARINING [imageHOGfromRoidbSample]: the\
        image field is not available for the above sample")
        return None
    img = cv2.imread(sample['image'])
    try:
        # scaleRawImage(img); maybe scale raw images differently in the future
        feature = HOGFromImage(img)
    except Exception as e:
        feature = None
        print(e)
        print('[imageHOGfromRoidbSample] hog failed @ path {}'.format(sample['image']))
    return feature

def HOGFromImage(image,rescale=True,orient=9, pix_per_cell=8,
                 spatial_size=(128,256), hist_bins=32,
                 cell_per_block=2):
    # hist_bins is *not* used

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if rescale: image = cv2.resize(image, spatial_size)

    hogFeatures =  get_hog_features(image[:,:], orient, 
                                    pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True)
    return hogFeatures

def appendHOGtoRoidb(roidb,size):
    print("="*100)
    print("appending the HOG field to Roidb")
    # HACK: skip to save space + time
    if size <= 1000: 
        addRoidbField(roidb,"hog",bboxHOGfromRoidbSample)
    addRoidbField(roidb,"hog_image",imageHOGfromRoidbSample)
    print("finished appending HOG")

def appendHOGtoRoidbDict(roidbDict,size):
    for roidb in roidbDict.values():
        appendHOGtoRoidb(roidb,size)

def getSampleWeight(y_test):
    weights = [0.0 for _ in cfg.DATASET_NAMES_ORDERED]
    for idx,ds in enumerate(cfg.DATASET_NAMES_ORDERED):
        weights[idx] = np.sum( y_test == idx )
    return weights

def make_confusion_matrix(model, X_test, y_test, clsToSet, normalize=True):

    y_pred = model.predict(X_test)    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)
    # we fixed the original ordering
    #cnf_matrix = switch_rows_cols(cnf_matrix,clsToSet,cfg.DATASET_NAMES_ORDERED)
    # todo Plot normalized confusion matrix  ----- NEED TO FIX CLASS NAMES DEPENDS ON PYROIDB
    return cnf_matrix


def plot_confusion_matrix(cm, classes, path_to_save, 
                          normalize=False,
                          cmap=plt.cm.Blues, show_plot = False,
                          vmin = 0, vmax = 100, title = None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    order = cfg.DATASET_NAMES_PAPER
    new_order = cfg.DATASET_NAMES_ORDERED
    
    fontdict = {'family':'monospace',
                'fontname':'Courier New',
                'size': 25
                }

    fig, ax = plt.subplots()

    print(cm)

    
    # todo: uncomement below
    cm = cm * 100
    cm = np.around(cm,0)

    # todo: remove me below
    _zeros = np.zeros(cm.shape)
    ax.imshow(_zeros, interpolation='nearest', cmap=cmap, vmin = vmin, vmax = vmax)

    # todo: uncomement below
    #ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin = vmin, vmax = vmax)
    classes = order
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,size="15",ha="right")
    plt.yticks(tick_marks, classes,size="15")

    fmt = '.0f'# if normalize else 'd'
    thresh = cm.max() / 2. * 1000000 # todo remove "10000000..."
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        t = ax.text(j, i+.2, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize="17")
    plt.subplots_adjust(hspace=0, wspace=0)
    if title != None:
        plt.title(title,fontdict=fontdict)
    plt.tight_layout()
    plt.savefig(path_to_save,transparent=True,bbox_inches='tight')
    if show_plot == True:
        plt.show()
    return cm

def classes_to_dict(classes):
    dict_classes = {}
    for i, name in enumerate(classes):
        dict_classes[name] = i
    return dict_classes

def switch_rows_cols(cm, classes, new_order):
    new_cm = np.copy(cm)
    for idx, nameA in enumerate(classes):
        for jdx, nameB in enumerate(classes):
            xVal = classes.index(new_order[idx])
            yVal = classes.index(new_order[jdx])
            new_cm[idx,jdx] = cm[xVal,yVal]
    # for i, name in enumerate(classes):
    #     new_cm[i,:] = cm[dict_classes[new_order[i]],:]

    # for i, name in enumerate(classes):
    #     new_cm[:,i] = cm[:, dict_classes[new_order[i]]]
    return new_cm

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True: # Call with two outputs if vis==True to visualize the HOG
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      # Otherwise call with one output
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to extract features from a list of images
def img_features(feature_image, feat_type, hist_bins, orient, 
                        pix_per_cell, cell_per_block):
    file_features = []
    # Call get_hog_features() with vis=False, feature_vec=True
    if feat_type == 'gray':
        feature_image = cv2.resize(feature_image, (32,32))
        feature_image = cv2.cvtColor(feature_image, cv2.COLOR_BGR2GRAY)
        file_features.append(feature_image.ravel())
    elif feat_type == 'color':
        feature_image = cv2.resize(feature_image, (32,32))
        file_features.append(feature_image.ravel())
    elif feat_type == 'hog':   
        # feature_image = cv2.cvtColor(feature_image, cv2.COLOR_LUV2RGB)
        feature_image = cv2.cvtColor(feature_image, cv2.COLOR_BGR2GRAY)
        feature_image = cv2.resize(feature_image, (128,256))
        #plt.imshow(feature_image)
        hog_features = get_hog_features(feature_image[:,:], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            #print 'hog', hog_features.shape
        # Append the new feature vector to the features list
        file_features.append(hog_features)
    return file_features

def extract_roidbDict_features(roidbDict, feat_type, clsToSet, calc_feat = False,
                               spatial_size=(32, 32),hist_bins=32, orient=9, 
                               pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to

    # Create a list to append feature vectors to
    features = []
    y = []
    l_feat = [[] for _ in range(len(cfg.DATASET_NAMES_ORDERED))]
    l_idx = [[] for _ in range(len(cfg.DATASET_NAMES_ORDERED))]

    for key,roidb in roidbDict.items():
        index = cfg.DATASET_NAMES_ORDERED.index(key)
        for i in range(len(roidb)):
            l_feat[index].append(roidb[i])

        
    errors = 0
    print('the length of the pyroidb is ', len(pyroidb))
    # Iterate through the list of images
    for i in range(len(pyroidb)):
        try:
            file_features = []
            inputs,target = pyroidb[i] # Read in each image one by one
            if calc_feat == True:
                img = img_features(inputs, feat_type, hist_bins, orient, pix_per_cell, cell_per_block)
                if i == 0:
                    print(img)
                l_feat[target].extend(img)
            else:
                l_feat[target].append(inputs)
            l_idx[target].append(i)
            y.append(target)
        except Exception as e:
            print(e,i)
            errors = errors + 1

    # now let's make each "sublist" a numpy array
    for idx in range(len(clsToSet)):
        l_feat[idx] = np.array(l_feat[idx])
        l_idx[idx] = np.array(l_idx[idx])

    if cfg.DEBUG:
        print("{} errors sorting pyroidb".format(errors))

    if cfg.DEBUG:
       for idx,name in enumerate(clsToSet):
           print("{}: {},{}".format(idx,name,len(l_idx[idx])))

    return l_feat, l_idx, y # Return list of feature vectors


def extract_pyroidb_features(pyroidb, feat_type, clsToSet, calc_feat = False, spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    y = []
    l_feat = [[] for _ in range(len(clsToSet))]
    l_idx = [[] for _ in range(len(clsToSet))]

    errors = 0
    print('the length of the pyroidb is ', len(pyroidb))
    # Iterate through the list of images
    for i in range(len(pyroidb)):
        try:
            file_features = []
            inputs,target = pyroidb[i] # Read in each image one by one
            if calc_feat == True:
                img = img_features(inputs, feat_type, hist_bins, orient, pix_per_cell, cell_per_block)
                if i == 0:
                    print(img)
                l_feat[target].extend(img)
            else:
                l_feat[target].append(HOGFromImage(inputs))
            l_idx[target].append(i)
            y.append(target)
        except Exception as e:
            print(e,i)
            errors = errors + 1

    # now let's make each "sublist" a numpy array
    for idx in range(len(clsToSet)):
        l_feat[idx] = np.array(l_feat[idx])
        l_idx[idx] = np.array(l_idx[idx])

    if cfg.DEBUG:
        print("{} errors sorting pyroidb".format(errors))

    if cfg.DEBUG:
       for idx,name in enumerate(clsToSet):
           print("{}: {},{}".format(idx,name,len(l_idx[idx])))

    return l_feat, l_idx, y # Return list of feature vectors

def splitFeatures(trainSize,testSize,inputFtrs):
    trainFtrs = inputFtrs[0:trainSize]
    testFtrs = inputFtrs[trainSize:trainSize + testSize]
    return trainFtrs,testFtrs

def split_data(train_size, test_size, ds_feat,l_idx, y, dsHasTest):
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    te_idx = []
    y = np.array(y)

    """
    Each class has a specific index determined by a file in "ymlConfigs"
    This means that a list of lists is totally cool for iterating over
    That is the key to compressing this code
    """
    clsToSet = cfg.DATASET_NAMES_ORDERED
    for idx in range(len(clsToSet)):

        feats = ds_feat[idx]
        ds_train_size = train_size
        ds_test_size = test_size
        # if there is not "test" set, split equally
        if not dsHasTest[idx]:
            ds_train_size = min(len(feats)/2,train_size)
            ds_test_size = ds_train_size
            
        print(feats[:ds_train_size].shape)
        x_train.append(feats[:ds_train_size])
        x_test.append(feats[ds_train_size:ds_train_size+ds_test_size])

        indicies = l_idx[idx]
        trIdx = indicies[:ds_train_size]
        teIdx = indicies[ds_train_size:ds_train_size+ds_test_size]
        print("<{}> dsHasTest: {} teIdx: {}".format(clsToSet[idx],dsHasTest[idx],len(teIdx)))

        y_train.extend(y[trIdx])

        te_idx.extend(teIdx)
        y_test.extend(y[teIdx])

    if cfg.DEBUG:
        for idx,name in enumerate(clsToSet):
            print("{}: {}, {}".format(idx,name,len(x_test[idx])))

    for elem in x_train:
        print(len(elem[0]))
        print(len(elem))
    X_train = np.vstack(x_train).astype(np.float64)
    X_test = np.vstack(x_test).astype(np.float64)
    X_idx = [{"idx":idx,"split":"train"} for idx in np.array(te_idx).astype(np.float64)]
    Y_train = np.array(y_train).astype(np.float64)
    Y_test = np.array(y_test).astype(np.float64)

    return X_train, X_test, Y_train, Y_test, X_idx

def dealwithKittiQuickly(feats_tr,y_tr,feats_te,y_te,train_size,test_size,tr_idx,te_idx):
    totalSize = len(feats_tr) + len(feats_te)
    xTr = []
    xTe = []
    yTr = []
    yTe = []
    idxTe = []
    if totalSize < (train_size + test_size):
        # our actual case

        xTr.append(feats_tr[:train_size])
        trIdx = l_idx_tr[idx][:train_size]
        yTr.extend(y_tr[trIdx])

        # we "break" if we have to do both....
        # addFromTest = train_size - len(xTr)
        # if addFromTest > 0:
        #     xTr.append(feats_te[:addFromTest])

        xTe.append(feats_te[:test_size])
        teIdx = l_idx_te[idx][:test_size]
        yTe.extend(y_te[teIdx])
        idxTe.extend([{"idx":idx,"split":"test"} for idx in teIdx])
        addFromTrain = test_size - len(xTe)
        if addFromTrain > 0:
            xTr.append(feats_tr[len(xTr):len(xTr)+addFromTest])
            teIdx = l_idx_tr[idx][len(xTr):len(xTr)+addFromTest]
            yTe.extend(y_tr[teIdx])
            idxTe.extend([{"idx":idx,"split":"train"} for idx in teIdx])
        
    return xTr,xTe,yTr,yTe,idxTe
    
def split_tr_te_data(ds_feat_tr,l_idx_tr,y_tr,
                     ds_feat_te,l_idx_te,y_te,
                     train_size, test_size, dsHasTest):

    x_train = []
    x_test = []

    y_train = []
    y_test = []

    te_idx = []

    print("y_tr looks like...")
    
    y_tr = np.array(y_tr)
    y_te = np.array(y_te)

    for idx,ds in enumerate(cfg.DATASET_NAMES_ORDERED):

        feats_tr = ds_feat_tr[idx]
        feats_te = ds_feat_te[idx]
        if ds == "kitti":

            x_tr_k,x_te_k,y_tr_k,y_te_k,te_idx_k = dealwithKittiQuickly(feats_tr,y_tr,
                                                                        feats_te,y_te,
                                                                        train_size,
                                                                        test_size,
                                                                        l_idx_tr,l_idx_te)
            print("kitti stats")
            print(len(x_tr_k),len(x_tr_k[0]))
            print(len(x_te_k),len(x_te_k[0]))
            print(len(y_tr_k),len(y_tr_k[0]))
            print(len(y_te_k),len(y_te_k[0]))
            print(len(x_train))
            print(len(x_test))
            print(len(y_train))
            print(len(y_test))
            x_train.append(x_tr_k)
            x_test.append(x_te_k)
            y_train.extend(y_tr_k)
            y_test.extend(y_te_k)
            te_idx.extend(te_idx_k)
            print(len(x_train))
            print(len(x_test))
            print(len(y_train))
            print(len(y_test))

        elif not dsHasTest[idx]:
            ds_train_size = min(len(feats_tr)/2,train_size)
            ds_test_size = ds_train_size

            x_train.append(feats_tr[:ds_train_size])
            x_test.append(feats_tr[ds_train_size:ds_train_size+ds_test_size])

            trIdx = l_idx_tr[idx][:ds_train_size]
            teIdx = l_idx_tr[idx][ds_train_size:ds_train_size+ds_test_size]

            y_train.extend(y_tr[trIdx])
            y_test.extend(y_te[teIdx])
            te_idx.extend([{"idx":idx,"split":"train"} for idx in teIdx])
        else:
            ds_train_size = train_size
            ds_test_size = test_size

            x_train.append(feats_tr[:ds_train_size])
            x_test.append(feats_te[:ds_test_size])

            trIdx = l_idx_tr[idx][:ds_train_size]
            teIdx = l_idx_te[idx][:ds_test_size]

            y_train.extend(y_tr[trIdx])
            y_test.extend(y_te[teIdx])

            if len(y_te[teIdx]) != len(feats_te[:ds_test_size]):
                print(ds,len(y_te[teIdx]),len(feats_te[:ds_test_size]))
                sys.exit()
            te_idx.extend([{"idx":idx,"split":"test"} for idx in teIdx])

    X_train = np.vstack(x_train).astype(np.float64)
    X_test = np.vstack(x_test).astype(np.float64)

    Y_train = np.array(y_train).astype(np.float64)
    Y_test = np.array(y_test).astype(np.float64)

    return X_train, X_test, Y_train, Y_test, te_idx


def scale_data(X_train, X_test):
    X_train_scaler = StandardScaler().fit(X_train)
    X_test_scaler = StandardScaler().fit(X_test)

    X_train_scaled = X_train_scaler.transform(X_train)
    X_test_scaled = X_test_scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def train_SVM(X_train, y_train):
    print('Feature vector length:', len(X_train[0]))
    svc = LinearSVC(loss='hinge', multi_class = 'ovr',class_weight='balanced') # Use a linear SVC 
    t=time.time() # Check the training time for the SVC
    print('start train')
    model_fit = svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    return model_fit

def test_acc(model, X_test, y_test):
    print('Test Accuracy of SVC = ', round(model.score(X_test, y_test), 4)) # Check the score of the SVC
    return

def boxToStr(box):
    return "{}_{}_{}_{}".format(
        box[0],box[1],box[2],box[3])

def findMaxRegions(topK,pyroidb,rawOutputs,y_test,testIndex,clsToSet):
    output_str = ""
    topKIndex = np.argsort(rawOutputs,axis=0)[-topK:,:]
    for idx,name in enumerate(clsToSet):
        print("{}: {}".format(idx,name))
        pyroidbIdx = testIndex[topKIndex[:,idx]]
        dsRawValues = rawOutputs[topKIndex[:,idx],idx]
        targets = y_test[topKIndex[:,idx]]
        for rowIdx,elemIdx in enumerate(pyroidbIdx):
            print(targets[rowIdx],targets[rowIdx])
            if targets[rowIdx] != targets[rowIdx]: continue
            elemIdx = int(elemIdx)
            sample,annoIndex = pyroidb.getSampleAtIndex(elemIdx)
            output_str += "{},{},{},{}\n".format(name,sample['image'],
                                                 boxToStr(sample['boxes'][annoIndex]),
                                                 dsRawValues[rowIdx])
    print(output_str)
    return output_str
