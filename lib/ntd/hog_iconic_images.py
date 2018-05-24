#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:32:08 2018

@author: zkapach
"""

#stuff from mixedDataReport.py
import _init_paths
from core.train import get_training_roidb
from core.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, loadDatasetIndexDict
from datasets.factory import get_repo_imdb
from datasets.ds_utils import load_mixture_set,print_each_size,computeTotalAnnosFromAnnoCount,cropImageToAnnoRegion
import os.path as osp
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys,os,cv2
# pytorch imports
from datasets.pytorch_roidb_loader import RoidbDataset



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

def extract_features(imgs, feat_type, spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file_p in imgs:
        file_features = []
        image = cv2.imread(file_p) # Read in each imageone by one
      #  print(type(image))
            # apply color conversion if other than 'RGB'
        if isinstance(image, np.ndarray) == True:
            feature_image = np.copy(image)      
            file_features = img_features(feature_image, feat_type, hist_bins, orient, pix_per_cell, cell_per_block, hog_channel)    
            features.append(np.concatenate(file_features))
#        feature_image=cv2.flip(feature_image,1) # Augment the dataset with flipped images
#        file_features = img_features(feature_image, hist_bins, orient, 
#                        pix_per_cell, cell_per_block, hog_channel)
#        #changed from cancatenate
#        features.append(file_features)
    return features # Return list of feature vectors








#more stuff from mixedDataReport.py in tools

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test loading a mixture dataset')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--setID', dest='setID',
                        help='which 8 digit ID to read from',
                        default='11111111', type=str)
    parser.add_argument('--repetition', dest='repetition',
                        help='which repetition to read from',
                        default='1', type=str)
    parser.add_argument('--size', dest='size',
                        help='which size to read from',
                        default=1000, type=int)
    parser.add_argument('--save', dest='save',
                        help='save some samples with bboxes visualized?',
                        action='store_true')
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_roidb(imdb_name):
    imdb = get_repo_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    imdb.set_proposal_method(cfg.TRAIN.OBJ_DET.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.OBJ_DET.PROPOSAL_METHOD)
    roidb = get_training_roidb(imdb)
    return imdb, roidb

def get_bbox_info(roidb,size):
    areas = np.zeros((size))
    widths = np.zeros((size))
    heights = np.zeros((size))
    actualSize = 0
    idx = 0
    for image in roidb:
        if image['flipped'] is True: continue
        bbox = image['boxes']
        for box in bbox:
            actualSize += 1
            widths[idx] = box[2] - box[0]
            heights[idx] = box[3] - box[1]
            assert widths[idx] >= 0,"widths[{}] = {}".format(idx,widths[idx])
            assert heights[idx] >= 0
            areas[idx] = widths[idx] * heights[idx]
            idx += 1
    return areas,widths,heights


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        np.random.seed(cfg.RNG_SEED)

    setID = args.setID
    repetition = args.repetition
    size = args.size
    
    roidb,annoCount = load_mixture_set(setID,repetition,size)
    numAnnos = computeTotalAnnosFromAnnoCount(annoCount)

    print("\n\n-=-=-=-=-=-=-=-=-\n\n")
    print("Report:\n\n")
    print("Mixture Dataset: {} {} {}\n\n".format(setID,repetition,size))

    print("number of images: {}".format(len(roidb)))
    print("number of annotations: {}".format(numAnnos))
    print("size of roidb in memory: {}kB".format(len(roidb) * sys.getsizeof(roidb[0])/1024.))
    print("example roidb:")
    for k,v in roidb[0].items():
        print("\t==> {},{}".format(k,type(v)))
        print("\t\t{}".format(v))

    print("computing bbox info...")
    areas, widths, heights = get_bbox_info(roidb,numAnnos)

    print("ave area: {} | std. area: {}".format(np.mean(areas),np.std(areas,dtype=np.float64)))
    print("ave width: {} | std. width: {}".format(np.mean(widths),np.std(widths,dtype=np.float64)))
    print("ave height: {} | std. height: {}".format(np.mean(heights),np.std(heights,dtype=np.float64)))
    prefix_path = cfg.IMDB_REPORT_OUTPUT_PATH
    if osp.exists(prefix_path) is False:
        os.makedirs(prefix_path)

    path = osp.join(prefix_path,"areas.dat")
    np.savetxt(path,areas,fmt='%.18e',delimiter=' ')
    path = osp.join(prefix_path,"widths.dat")
    np.savetxt(path,widths,fmt='%.18e',delimiter=' ')
    path = osp.join(prefix_path,"heights.dat")
    np.savetxt(path,heights,fmt='%.18e',delimiter=' ')

        
    print("-=-=-=-=-=-")

    clsToSet = loadDatasetIndexDict()

    print("as pytorch friendly ")

    pyroidb = RoidbDataset(roidb,[1,2,3,4,5,6,7,8],loader=cv2.imread,transform=cropImageToAnnoRegion)

    if args.save:
       print("save 30 cropped annos in output folder.")
       saveDir = "./output/mixedDataReport/"
       if not osp.exists(saveDir):
           print("making directory: {}".format(saveDir))
           os.makedirs(saveDir)

       for i in range(30):
           cls = roidb[i]['set']
           ds = clsToSet[cls]
           fn = osp.join(saveDir,"{}_{}.jpg".format(i,ds))
           print(fn)
           cv2.imwrite(fn,pyroidb[i][0])

    print(pyroidb) 





# #LOAD IN DATA for Helps computer
# train_cam = '/home/party/labeling-party/xa/'
# train_COCO = '/var/data/coco/images/train2014/'
# train_INRIA = '/var/data/inria/INRIAPerson/Train/'
# train_caltech = '/var/data/caltech_pedestrian/CAL2009/images/'
# train_ImageNet = '/var/data/ilsvrc/ILSVRC/ILSVRC13/ILSVRC_DET_train/'
# train_Pascal = '/var/data/pascal_voc/VOCdevkit/VOC2012/JPEGImages/'
# train_Sun = '/var/data/sun/SUN2012/Images/'
# train_kitti = '/srv/sdb1/image_team/kitti/KITTIdevkit/KITTI2013/image_2/'


# #for helps computer
# coco = glob.glob(train_COCO + '/**/*.png', recursive = True) + (glob.glob(train_COCO + '/**/*.jpg', recursive = True))#+ (glob.glob(train_COCO + '/**/*.bmp', recursive = True))
# caltech = glob.glob(train_caltech + '/**/*.png', recursive = True) + (glob.glob(train_caltech + '/**/*.jpg', recursive = True))#+ (glob.glob(train_caltech + '/**/*.bmp', recursive = True))
# imagenet = glob.glob(train_ImageNet + '/**/*.JPEG', recursive = True) + (glob.glob(train_ImageNet + '/**/*.jpg', recursive = True))#+ (glob.glob(train_ImageNet + '/**/*.bmp', recursive = True))
# pascal = glob.glob(train_Pascal + '/**/*.png', recursive = True) + (glob.glob(train_Pascal + '/**/*.jpg', recursive = True))#+ (glob.glob(train_Pascal + '/**/*.bmp', recursive = True))

# cam2_temp = glob.glob(train_cam + '/**/*.png', recursive = True)
# cam2 = [f for f,j in zip(cam2_temp, range(len(cam2_temp))) if (j%100)==0]
# inria = glob.glob(train_INRIA + '/**/*.png', recursive = True) + (glob.glob(train_INRIA + '/**/*.jpg', recursive = True))
# sun = glob.glob(train_Sun + '/**/*.jpg', recursive = True)
# kitti = glob.glob(train_kitti + '/**/*.png', recursive = True)
# #'gray', 'color',
# model = ['gray', 'color','hog']
# results = []
# for j in model:
#    # for i in range(10):
#     t3 = time.time()
#     shuffle(coco)
#     shuffle(caltech)
#     shuffle(imagenet)
#     shuffle(pascal)
#     shuffle(cam2)
#     shuffle(inria)
#     shuffle(sun)
#     shuffle(kitti)
    
#     print('len of coco', len(coco))
#     print('len of caltech', len(caltech))
#     print('len of imagenet', len(imagenet))
#     print('len of pascal' ,len(pascal))
#     print('len of cam2', len(cam2))
#     print('len of inria', len(inria))
#     print('len of sun', len(sun))
#     print('len of kitti', len(kitti))
    
#     dataset_size_train = 1000
#     dataset_size_test = 300
#     dataset_total = dataset_size_train+dataset_size_test
#     mix_size_train = math.floor(dataset_size_train/8)
#     mix_size_test = math.floor(dataset_size_test/8)
#     dataset_total2 = dataset_total + mix_size_train
    
    
#     coco1 = coco[0:(dataset_size_train+dataset_size_test)]
#     caltech1 = caltech[0:(dataset_size_train+dataset_size_test)]
#     imagenet1 = imagenet[0:(dataset_size_train+dataset_size_test)]
#     pascal1 = pascal[0:(dataset_size_train+dataset_size_test)]
#     cam21 = cam2[0:(dataset_size_train+dataset_size_test)]
#     inria1 = inria[0:(dataset_size_train+dataset_size_test)]
#     sun1 = sun[0:(dataset_size_train+dataset_size_test)]
#     kitti1 = kitti[0:(dataset_size_train+dataset_size_test)]
#     mix1_train_images = coco[(dataset_total):(dataset_total+ mix_size_train)] +caltech[(dataset_total):(dataset_total+ mix_size_train)] +imagenet[(dataset_total):(dataset_total+ mix_size_train)] +pascal[(dataset_total):(dataset_total+ mix_size_train)] + cam2[(dataset_total):(dataset_total+ mix_size_train)] + inria[(dataset_total):(dataset_total+ mix_size_train)] + sun[(dataset_total):(dataset_total+ mix_size_train)] + kitti[(dataset_total):(dataset_total+ mix_size_train)]
#     mix1_test_images = coco[(dataset_total2):(dataset_total2+ mix_size_test)] +caltech[(dataset_total2):(dataset_total2+ mix_size_test)] +imagenet[(dataset_total2):(dataset_total2+ mix_size_test)] +pascal[(dataset_total2):(dataset_total2+ mix_size_test)] + cam2[(dataset_total2):(dataset_total2+ mix_size_test)] + inria[(dataset_total2):(dataset_total2+ mix_size_test)] + sun[(dataset_total2):(dataset_total2+ mix_size_test)] + kitti[(dataset_total2):(dataset_total2+ mix_size_test)]
    
#     print('len after cut')
#     print('len of coco', len(coco1))
#     print('len of caltech', len(caltech1))
#     print('len of imagenet', len(imagenet1))
#     print('len of pascal' ,len(pascal1))
#     print('len of cam2', len(cam21))
#     print('len of inria', len(inria1))
#     print('len of sun', len(sun1))
#     print('len of kitti', len(kitti1))
    
#     orient = 8  # HOG orientations
#     pix_per_cell = 8 # HOG pixels per cell
#     cell_per_block = 2 # HOG cells per block
#     hog_channel = 0 # Can be 0, 1, 2, or "ALL"
#     spatial_size = (16, 16) # Spatial binning dimensions
#     hist_bins = 32    # Number of histogram bins
    
#     coco_feat = extract_features(coco1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
#     caltech_feat = extract_features(caltech1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
#     imagenet_feat = extract_features(imagenet1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel) 
#     pascal_feat = extract_features(pascal1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
#     cam2_feat = extract_features(cam21, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
#     inria_feat = extract_features(inria1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
#     sun_feat = extract_features(sun1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
#     kitti_feat = extract_features(kitti1, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
#     mix_train = extract_features(mix1_train_images, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
#     mix_test = extract_features(mix1_test_images, j,spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    
    
#     coco_train = coco_feat[0:dataset_size_train]
#     coco_test = coco_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
#     caltech_train = caltech_feat[0:dataset_size_train]
#     caltech_test = caltech_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
#     imagenet_train = imagenet_feat[0:dataset_size_train]
#     imagenet_test = imagenet_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
#     pascal_train = pascal_feat[0:dataset_size_train]
#     pascal_test = pascal_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
#     cam2_train = cam2_feat[0:dataset_size_train]
#     cam2_test = cam2_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
#     inria_train = inria_feat[0:dataset_size_train]
#     inria_test = inria_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
#     sun_train = sun_feat[0:dataset_size_train]
#     sun_test = sun_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
#     kitti_train = kitti_feat[0:dataset_size_train]
#     kitti_test = kitti_feat[dataset_size_train:(dataset_size_train+dataset_size_test)]
    
    
#     print('len after feat extract and cut')
#     print('len of coco', len(coco_train), len(coco_test))
#     print('len of caltech', len(caltech_train), len(caltech_test))
#     print('len of imagenet', len(imagenet_train), len(imagenet_test))
#     print('len of pascal' ,len(pascal_train), len(pascal_test))
#     print('len of cam2', len(cam2_train), len(cam2_test))
#     print('len of inria', len(inria_train), len(inria_test))
#     print('len of sun', len(sun_train), len(sun_test))
#     print('len of kitti', len(kitti_train), len(kitti_test))
#     print('len of mix', len(mix_train), len(mix_test))
#     X_train_m = np.vstack((coco_train, caltech_train, imagenet_train, pascal_train, cam2_train, inria_train, sun_train, kitti_train, mix_train)).astype(np.float64)
#     X_test_m = np.vstack((coco_test, caltech_test, imagenet_test, pascal_test, cam2_test, inria_test, sun_test, kitti_test, mix_test)).astype(np.float64)
#     X_train_scaler = StandardScaler().fit(X_train_m)
#     X_test_scaler = StandardScaler().fit(X_test_m)
    
    
#     X_train_scaled = X_train_scaler.transform(X_train_m)
#     X_test_scaled = X_test_scaler.transform(X_test_m)
    
#     y_train = np.hstack((np.ones(len(coco_train)), np.full(len(caltech_train), 2), np.full(len(imagenet_train), 3), np.full(len(pascal_train), 4), np.full(len(cam2_train), 5), np.full(len(inria_train), 6), np.full(len(sun_train), 7), np.full(len(kitti_train), 8), np.full(len(mix_train), 9)))
#     y_test = np.hstack((np.ones(len(coco_test)), np.full(len(caltech_test), 2), np.full(len(imagenet_test), 3), np.full(len(pascal_test), 4), np.full(len(cam2_test), 5), np.full(len(inria_test), 6), np.full(len(sun_test), 7), np.full(len(kitti_test), 8), np.full(len(mix_test), 9)))
    
    
#     print('Using:',orient,'orientations', pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
#     print('Feature vector length:', len(X_train_scaled[0]))
#     svc = LinearSVC(loss='hinge', multi_class = 'ovr') # Use a linear SVC 
#     t=time.time() # Check the training time for the SVC
# #        n_estimators = 10
#     print('start train')
# #        model = BaggingClassifier(svc, max_samples=1.0 / n_estimators, n_estimators=n_estimators) # Train the classifier
#     model_fit = svc.fit(X_train_scaled, y_train)
#     t2 = time.time()
#     print(round(t2-t, 2), 'Seconds to train SVC...')
#     print('Test Accuracy of SVC = ', round(model_fit.score(X_test_scaled, y_test), 4)) # Check the score of the SVC
#     results.append(round(model_fit.score(X_test_scaled, y_test), 4))
   
#     t4 = time.time()
#     print(round(t4-t3, 2), 'Seconds to run')

#     # if i == 0:
                        
#     #     ############# FOR CONFUSION MATRIX #################################
        
#     #     y_pred = model_fit.predict(X_test_scaled)
        
#     #     # Compute confusion matrix
#     #     cnf_matrix = confusion_matrix(y_test, y_pred)
#     #     np.set_printoptions(precision=2)
        
#     #     # Plot normalized confusion matrix
#     #     class_names = ('COCO', 'Caltech', 'ImageNet', 'Pascal', 'Cam2', 'INRIA', 'Sun', 'Kitti', 'Mixed')
#     #     plt.figure()
#     #     plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
        
#     #     plt.show()
        
# print('results are', results)
# input()
# #    
# ############## FOR CONFUSION MATRIX #################################
# #
# #y_pred = model_fit.predict(X_test_scaled)
# #
# ## Compute confusion matrix
# #cnf_matrix = confusion_matrix(y_test, y_pred)
# #np.set_printoptions(precision=2)
# #
# ## Plot normalized confusion matrix
# #class_names = ('COCO', 'Caltech', 'ImageNet', 'Pascal', 'Cam2', 'INRIA', 'Sun')
# #plt.figure()
# #plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
# #                      title='Normalized confusion matrix')
# #
# #plt.show()
# #input("Press enter to continue")