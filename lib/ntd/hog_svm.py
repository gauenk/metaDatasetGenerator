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
from random import shuffle
from sklearn.ensemble import BaggingClassifier
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')



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

def extract_pyroidb_features(pyroidb, feat_type, spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    y = []
    coco_feat = []
    voc_feat = []
    imageNet_feat = []
    cam2_feat = []
    inria_feat = []
    caltech_feat = []
    sun_feat = []
    kitti_feat = []

    coco_idx = []
    voc_idx = []
    imageNet_idx = []
    cam2_idx = []
    inria_idx = []
    caltech_idx = []
    sun_idx = []
    kitti_idx = []

    # i = 0
    errors = 0
    print('the length of the pyroidb is ', len(pyroidb))
    # Iterate through the list of images
    for i in range(0, len(pyroidb)):
        # print(pyroidb[i][1])
    # for file_p in pyroidb:
        try:
            file_features = []
            image = pyroidb[i][0] # Read in each imageone by one
          #  print(type(image))
                # apply color conversion if other than 'RGB'
            if isinstance(image, np.ndarray) == True: 
                feature_image = np.copy(image)      
                file_features = img_features(feature_image, feat_type, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel)    
                # features.append(np.concatenate(file_features))
                y.append(pyroidb[i][1])

                # if mixed we need to do something like this
            if pyroidb[i][1] == 1:
                coco_feat.append(np.concatenate(file_features))
                coco_idx.append([i])
            elif pyroidb[i][1] == 2:
                voc_feat.append(np.concatenate(file_features))
                voc_idx.append([i])
            elif pyroidb[i][1] == 3:    
                imageNet_feat.append(np.concatenate(file_features))
                imageNet_idx.append([i])
            elif pyroidb[i][1] == 4:   
                cam2_feat.append(np.concatenate(file_features))
                cam2_idx.append([i])
            elif pyroidb[i][1] == 5:    
                inria_feat.append(np.concatenate(file_features))
                inria_idx.append([i])
            elif pyroidb[i][1] == 6:    
                caltech_feat.append(np.concatenate(file_features))
                caltech_idx.append([i])
            elif pyroidb[i][1] == 7:    
                sun_feat.append(np.concatenate(file_features))
                sun_idx.append([i])
            elif pyroidb[i][1] == 8:    
                kitti_feat.append(np.concatenate(file_features))
                kitti_idx.append([i])
        except: 
            errors = errors + 1
    print('1')
    print(coco_idx)
    print('2')

    print(voc_idx)
    print('3')
    print(imageNet_idx)
    print('4')
    print(cam2_idx)
    print('5')
    print(inria_idx)
    print('6')
    print(caltech_idx)
    print('7')
    print(sun_idx)
    print('8')
    print(kitti_idx)
    # X = np.vstack((coco_feat, voc_feat, imageNet_feat, cam2_feat, inria_feat, caltech_feat, sun_feat, kitti_feat)).astype(np.float64)
    X_idx = np.vstack((coco_idx, voc_idx, imageNet_idx, cam2_idx, inria_idx, caltech_idx, sun_idx, kitti_idx)).astype(np.float64)

    y.sort()
    print('number of errors = ', errors)
#        feature_image=cv2.flip(feature_image,1) # Augment the dataset with flipped images
#        file_features = img_features(feature_image, hist_bins, orient, 
#                        pix_per_cell, cell_per_block, hog_channel)
#        #changed from cancatenate
#        features.append(file_features)
    return coco_feat, voc_feat, imageNet_feat, cam2_feat, inria_feat, caltech_feat, sun_feat, kitti_feat, y, X_idx # Return list of feature vectors



def split_data(train_size, test_size, coco_feat, voc_feat, imagenet_feat, cam2_feat, inria_feat, caltech_feat, sun_feat, kitti_feat, X_idx, y):
    y_train = []
    y_test = []
    # coco_idx = []
    # voc_idx = []
    # imageNet_idx = []
    # cam2_idx = []
    # inria_idx = []
    # caltech_idx = []
    # sun_idx = []
    # kitti_idx = []


    coco_train = coco_feat[0:train_size]
    y_train.append(y[0:train_size]) 
    idx_loc = len(coco_feat)

    voc_train = voc_feat[0:train_size]
    y_train.append(y[idx_loc:(idx_loc + train_size)]) 
    idx_loc = idx_loc + len(voc_feat)

    imagenet_train = imagenet_feat[0:train_size]
    y_train.append(y[idx_loc:(idx_loc + train_size)]) 
    idx_loc = idx_loc + len(imagenet_feat)

    cam2_train = cam2_feat[0:train_size]
    y_train.append(y[idx_loc:(idx_loc + train_size)]) 
    idx_loc = idx_loc + len(cam2_feat)

    inria_train = inria_feat[0:train_size]
    y_train.append(y[idx_loc:(idx_loc + train_size)]) 
    idx_loc = idx_loc + len(inria_feat)

    caltech_train = caltech_feat[0:train_size]
    y_train.append(y[idx_loc:(idx_loc + train_size)]) 
    idx_loc = idx_loc + len(caltech_feat)

    sun_train = sun_feat[0:train_size]
    y_train.append(y[idx_loc:(idx_loc + train_size)]) 
    idx_loc = idx_loc + len(sun_feat)

    kitti_train = kitti_feat[0:train_size]
    y_train.append(y[idx_loc:(idx_loc + train_size)]) 


    try:
        coco_test = coco_feat[train_size: train_size + test_size]
        coco_idx = X_idx[train_size: train_size + test_size]
        y_test.append(y[(train_size):(train_size+test_size)]) 

    except: 
        coco_test = coco_feat[train_size: train_size + len(coco_feat)]
        coco_idx = X_idx[train_size: train_size + len(coco_feat)]
        y_test.append(y[(train_size):(train_size+len(coco_feat))]) 

   
    idx_loc = len(coco_feat)
    
    try: 
        voc_test = voc_feat[train_size: train_size + test_size]
        voc_idx = X_idx[(idx_loc+train_size): (idx_loc + train_size + test_size)]
        y_test.append(y[(idx_loc+train_size):(idx_loc + train_size+test_size)]) 

    except:
        voc_test = voc_feat[train_size: train_size + len(voc_feat)]
        voc_idx = X_idx[(idx_loc+train_size): (idx_loc + train_size + len(voc_feat))]
        y_test.append(y[(idx_loc+train_size):(idx_loc + train_size+len(voc_feat))]) 


    idx_loc = idx_loc + len(voc_feat)
    
    try:
        imagenet_test = imagenet_feat[train_size: train_size + test_size]
        imagenet_idx = X_idx[(idx_loc+train_size): (idx_loc + train_size + test_size)]
        y_test.append(y[(idx_loc+train_size):(idx_loc + train_size+test_size)]) 
    except: 
        imagenet_test = imagenet_feat[train_size: train_size + len(imagenet_feat)]
        imagenet_idx = X_idx[(idx_loc+train_size): (idx_loc + train_size + len(imagenet_feat))]
        y_test.append(y[(idx_loc+train_size):(idx_loc + train_size+len(imagenet_feat))]) 

    idx_loc = idx_loc + len(imagenet_feat)

    try: 
        cam2_test = cam2_feat[train_size: train_size + test_size]
        cam2_idx = X_idx[(idx_loc+train_size): (idx_loc + train_size + test_size)]
        y_test.append(y[(idx_loc+train_size):(idx_loc + train_size+test_size)]) 
    except: 
        cam2_test = cam2_feat[train_size: train_size + len(cam2_feat)]
        cam2_idx = X_idx[(idx_loc+train_size): (idx_loc + train_size + len(cam2_feat))]
        y_test.append(y[(idx_loc+train_size):(idx_loc + train_size+len(cam2_feat))]) 

    idx_loc = idx_loc + len(cam2_feat)

    try: 
        inria_test = inria_feat[train_size: train_size + test_size]
        inria_idx = X_idx[(idx_loc+train_size): (idx_loc + train_size + test_size)]
        y_test.append(y[(idx_loc+train_size):(idx_loc + train_size+test_size)]) 
    except:
        inria_test = inria_feat[train_size: train_size + len(inria_feat)]
        inria_idx = X_idx[(idx_loc+train_size): (idx_loc + train_size + len(inria_feat))]
        y_test.append(y[(idx_loc+train_size):(idx_loc + train_size+len(inria_feat))]) 

    idx_loc = idx_loc + len(inria_feat)
    
    try:
        caltech_test = caltech_feat[train_size: train_size + test_size]
        caltech_idx = X_idx[(idx_loc+train_size): (idx_loc + train_size + test_size)]
        y_test.append(y[(idx_loc+train_size):(idx_loc + train_size+test_size)]) 
    except:
        caltech_test = caltech_feat[train_size: train_size + len(caltech_feat)]
        caltech_idx = X_idx[(idx_loc+train_size): (idx_loc + train_size + len(caltech_feat))]
        y_test.append(y[(idx_loc+train_size):(idx_loc + train_size+len(caltech_feat))]) 

    idx_loc = idx_loc + len(caltech_feat)

    try: 
        sun_test = sun_feat[train_size: train_size + test_size]
        sun_idx = X_idx[(idx_loc+train_size): (idx_loc + train_size + test_size)]
        y_test.append(y[(idx_loc+train_size):(idx_loc + train_size+test_size)]) 
    except:
        sun_test = sun_feat[train_size: train_size + len(sun_feat)]
        sun_idx = X_idx[(idx_loc+train_size): (idx_loc + train_size + len(sun_feat))]
        y_test.append(y[(idx_loc+train_size):(idx_loc + train_size+len(sun_feat))]) 

    idx_loc = idx_loc + len(sun_feat)

    try:
        kitti_test = kitti_feat[train_size: train_size + test_size]
        kitti_idx = X_idx[(idx_loc+train_size): (idx_loc + train_size + test_size)]
        y_test.append(y[(idx_loc+train_size):(idx_loc + train_size+test_size)]) 
    except:
        kitti_test = kitti_feat[train_size: train_size + len(kitti_feat)]
        kitti_idx = X_idx[(idx_loc+train_size): (idx_loc + train_size + len(kitti_feat))]
        y_test.append(y[(idx_loc+train_size):(idx_loc + train_size+len(kitti_feat))]) 

    print('the coco idx is ', type(coco_idx), type(coco_idx[0]))
    print(type(coco_idx[0][0]))
    print('1')
    print(coco_test)
    print('2')

    print(voc_test)
    print('3')
    print(imagenet_test)
    print('4')
    print(cam2_test)
    print('5')
    print(inria_test)
    print('6')
    print(caltech_test)
    print('7')
    print(sun_test)
    print('8')
    print(kitti_test)
    X_train = np.vstack((coco_train, voc_train, imagenet_train, cam2_train, inria_train, caltech_train, sun_train, kitti_train)).astype(np.float64)
    X_test = np.vstack((coco_test, voc_test, imagenet_test, cam2_test, inria_test, caltech_test, sun_test, kitti_test)).astype(np.float64)
    X_idx = np.vstack((coco_idx, voc_idx, imageNet_idx, cam2_idx, inria_idx, caltech_idx, sun_idx, kitti_idx)).astype(np.float64)

    return X_train, X_test, y_train, y_test, X_idx



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

