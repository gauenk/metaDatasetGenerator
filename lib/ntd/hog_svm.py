#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:32:08 2018

@author: zkapach
"""
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import itertools
import os
import glob
import time
import math
from core.config import cfg
from random import shuffle
from sklearn.ensemble import BaggingClassifier
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.metrics import confusion_matrix
from datasets.ds_utils import cropImageToAnnoRegion,addRoidbField,clean_box
import warnings
warnings.filterwarnings('ignore')

# cfg.DEBUG = True

def HOGfromRoidbSample(sample,orient=9, pix_per_cell=8,
                       cell_per_block=2, hog_channel=0):
    features = []
    img = cv2.imread(sample['image'])
    for box in sample['boxes']:
        clean_box(box,sample)
        cimg = cropImageToAnnoRegion(img,box)
        feature_image = np.copy(cimg)      
        try:
            features.append(HOGFromImage(feature_image))
        except:
            print(sample)
            return None
    return features

def HOGFromImage(image,orient=9, pix_per_cell=8,
                 spatial_size=(128,256), hist_bins=32,
                 cell_per_block=2):
    # hist_bins is *not* used

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, spatial_size)
    hogFeatures =  get_hog_features(image[:,:], orient, 
                                    pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True)
    return hogFeatures

def appendHOGtoRoidb(roidb):
    print("="*100)
    print("appending the HOG field to Roidb")
    addRoidbField(roidb,"hog",HOGfromRoidbSample)
    print("finished appending HOG")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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
                        pix_per_cell, cell_per_block, hog_channel):
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
        #NEED TO FIX FIRST IF
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
        else:
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

def extract_pyroidb_features(pyroidb, feat_type, clsToSet, spatial_size=(32, 32),
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
            inputs,target = pyroidb[i] # Read in each imageone by one
            l_feat[target].append(inputs)
            l_idx[target].append(i)
            y.append(target)
        except Exception as e:
            print(e)
            errors = errors + 1

    # now let's make each "sublist" a numpy array
    for idx in range(len(clsToSet)):
        l_feat[idx] = np.array(l_feat[idx])
        l_idx[idx] = np.array(l_idx[idx])

    if cfg.DEBUG:
        print("{} errors sorting pyroidb".format(errors))

    if cfg.DEBUG:
       for idx,name in enumerate(clsToSet):
           print("{}: {}".format(idx,name))
           print(l_idx[idx])

    return l_feat, l_idx, y # Return list of feature vectors

def splitFeatures(trainSize,testSize,inputFtrs):
    trainFtrs = inputFtrs[0:trainSize]
    testFtrs = inputFtrs[trainSize:trainSize + testSize]
    return trainFtrs,testFtrs

def split_data(train_size, test_size, l_feat,l_idx, y, clsToSet):
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    te_idx = [ [] for _ in range(len(clsToSet)) ]
    y = np.array(y)

    """
    @zohar
    Each class has a specific index determined by a file in "ymlConfigs"
    This means that a list of lists is totally cool for iterating over
    That is the key to compressing this code
    """
    for idx in range(len(clsToSet)):

        feats = l_feat[idx]
        x_train.append(feats[:train_size])
        x_test.append(feats[train_size:train_size+test_size])

        indicies = l_idx[idx]
        trIdx = indicies[:train_size]
        teIdx = indicies[train_size:train_size+test_size]
        te_idx[idx] = teIdx
        y_train.extend(y[trIdx])
        y_test.extend(y[teIdx])

    if cfg.DEBUG:
        for idx,name in enumerate(clsToSet):
            print("{}: {}".format(idx,name))
            print(x_test[idx])

    X_train = np.vstack(x_train).astype(np.float64)
    X_test = np.vstack(x_test).astype(np.float64)
    Y_train = np.array(y_train).astype(np.float64)
    Y_test = np.array(y_test).astype(np.float64)
    X_idx = np.vstack(te_idx).astype(np.float64)

    return X_train, X_test, Y_train, Y_test, X_idx

def scale_data(X_train, X_test):
    X_train_scaler = StandardScaler().fit(X_train)
    X_test_scaler = StandardScaler().fit(X_test)

    X_train_scaled = X_train_scaler.transform(X_train)
    X_test_scaled = X_test_scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def train_SVM(X_train, y_train):
    print('Feature vector length:', len(X_train[0]))
    svc = LinearSVC(loss='hinge', multi_class = 'ovr') # Use a linear SVC 
    t=time.time() # Check the training time for the SVC
    print('start train')
    model_fit = svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    return model_fit


def make_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    
    # Plot normalized confusion matrix  ----- NEED TO FIX CLASS NAMES DEPENDS ON PYROIDB
    class_names = ('COCO', 'Caltech', 'ImageNet', 'Pascal', 'Cam2', 'INRIA', 'Sun', 'Kitti', 'Mixed')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
    plt.show()
    return

def test_acc(model, X_test, y_test):
    print('Test Accuracy of SVC = ', round(model.score(X_test, y_test), 4)) # Check the score of the SVC
    return

def findMaxRegions(pyroidb,rawOutputs,l_idx):
    # TODO: write this function
    pass
    
