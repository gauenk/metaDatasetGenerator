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

def extract_features(pyroidb, feat_type, spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    y = []
    # Iterate through the list of images
    for file_p in pyroidb:
        file_features = []
        image = file_p[0] # Read in each imageone by one
      #  print(type(image))
            # apply color conversion if other than 'RGB'
        if isinstance(image, np.ndarray) == True:
            feature_image = np.copy(image)      
            file_features = img_features(feature_image, feat_type, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel)    
            features.append(np.concatenate(file_features))
            y.append(file_p[1])

        # if mixed we need to do something like this
        # if file_p[1] == 1:
        #     y.append(1)
        # else if file_p[1] == 2:
        #     y.append(2)
        # else if file_p[1] == 3:    
        #     y.append(3)
        # else if file_p[1] == 4:   
        #     y.append(4)
        # else if file_p[1] == 5:    
        #     y.append(5)
        # else if file_p[1] == 6:    
        #     y.append(6)
        # else if file_p[1] == 7:    
        #     y.append(7)
        # else if file_p[1] == 8:    
        #     y.append(8)

        
#        feature_image=cv2.flip(feature_image,1) # Augment the dataset with flipped images
#        file_features = img_features(feature_image, hist_bins, orient, 
#                        pix_per_cell, cell_per_block, hog_channel)
#        #changed from cancatenate
#        features.append(file_features)
    return features # Return list of feature vectors





def split_data()


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
    
    coco_train = coco_feat[0:dataset_size_train]
    coco_test = coco_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    caltech_train = caltech_feat[0:dataset_size_train]
    caltech_test = caltech_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    imagenet_train = imagenet_feat[0:dataset_size_train]
    imagenet_test = imagenet_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    pascal_train = pascal_feat[0:dataset_size_train]
    pascal_test = pascal_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    cam2_train = cam2_feat[0:dataset_size_train]
    cam2_test = cam2_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    inria_train = inria_feat[0:dataset_size_train]
    inria_test = inria_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    sun_train = sun_feat[0:dataset_size_train]
    sun_test = sun_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    kitti_train = kitti_feat[0:dataset_size_train]
    kitti_test = kitti_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    
    
    print('len after feat extract and cut')
    print('len of coco', len(coco_train), len(coco_test))
    print('len of caltech', len(caltech_train), len(caltech_test))
    print('len of imagenet', len(imagenet_train), len(imagenet_test))
    print('len of pascal' ,len(pascal_train), len(pascal_test))
    print('len of cam2', len(cam2_train), len(cam2_test))
    print('len of inria', len(inria_train), len(inria_test))
    print('len of sun', len(sun_train), len(sun_test))
    print('len of kitti', len(kitti_train), len(kitti_test))
    print('len of mix', len(mix_train), len(mix_test))
    
    X_train_m = np.vstack((coco_train, caltech_train, imagenet_train, pascal_train, cam2_train, inria_train, sun_train, kitti_train, mix_train)).astype(np.float64)
    X_test_m = np.vstack((coco_test, caltech_test, imagenet_test, pascal_test, cam2_test, inria_test, sun_test, kitti_test, mix_test)).astype(np.float64)

    y_train = np.hstack((np.ones(len(coco_train)), np.full(len(caltech_train), 2), np.full(len(imagenet_train), 3), np.full(len(pascal_train), 4), np.full(len(cam2_train), 5), np.full(len(inria_train), 6), np.full(len(sun_train), 7), np.full(len(kitti_train), 8), np.full(len(mix_train), 9)))
    y_test = np.hstack((np.ones(len(coco_test)), np.full(len(caltech_test), 2), np.full(len(imagenet_test), 3), np.full(len(pascal_test), 4), np.full(len(cam2_test), 5), np.full(len(inria_test), 6), np.full(len(sun_test), 7), np.full(len(kitti_test), 8), np.full(len(mix_test), 9)))
    
    

    print('Test Accuracy of SVC = ', round(model_fit.score(X_test_scaled, y_test), 4)) # Check the score of the SVC
    results.append(round(model_fit.score(X_test_scaled, y_test), 4))
   
    t4 = time.time()
    print(round(t4-t3, 2), 'Seconds to run')


