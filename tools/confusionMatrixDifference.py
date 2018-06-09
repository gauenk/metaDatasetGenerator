#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:32:08 2018

@author: zkapach
"""
import matplotlib
matplotlib.use("Agg")


import _init_paths
from core.train import get_training_roidb
from core.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, loadDatasetIndexDict,iconicImagesFileFormat
from datasets.factory import get_repo_imdb
from datasets.ds_utils import load_mixture_set,print_each_size,computeTotalAnnosFromAnnoCount,cropImageToAnnoRegion,roidbSampleHOG,roidbSampleImage
import os.path as osp
import datasets.imdb
import argparse
import pprint
import numpy as np
import matplotlib.pyplot as plt
import sys,os,cv2,pickle
# pytorch imports
from datasets.pytorch_roidb_loader import RoidbDataset
from numpy import transpose as npt
from ntd.hog_svm import plot_confusion_matrix, extract_pyroidb_features,appendHOGtoRoidb,split_data, scale_data,train_SVM,findMaxRegions, make_confusion_matrix
from utils.misc import *

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
    parser.add_argument('--repeat', dest='repeat',
                        help='which repeat to read from',
                        default='1', type=str)
    parser.add_argument('--size', dest='size',
                        help='which size to read from',
                        default=250, type=int)
    parser.add_argument('--save', dest='save',
                        help='save some samples with bboxes visualized?',
                        action='store_true')
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--modelRaw', dest='modelRaw',
                        help='give the path to a fit model',
                        default=None, type=str)
    parser.add_argument('--modelCropped', dest='modelCropped',
                        help='give the path to a fit model',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

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

def roidbToFeatures(roidb,pyloader=roidbSampleHOG,calcHog=False):
    pyroidb = RoidbDataset(roidb,[0,1,2,3,4,5,6,7],
                           loader=pyloader,
                           transform=None)

    l_feat,l_idx,y = extract_pyroidb_features(pyroidb, 'hog', clsToSet, calc_feat = calcHog, \
                                              spatial_size=(32, 32),hist_bins=32, \
                                              orient=9, pix_per_cell=8, cell_per_block=2, \
                                              hog_channel=0)
    return l_feat,l_idx,y

def mangleTestingData(l_feat_te,l_idx_te,y_te,X_test,y_test,X_idx,test_size):
    # replace the X_test for each match of y_test
    yIndicies = {}
    for idx,setID in enumerate(y_te):
        if setID not in yIndicies.keys():
            yIndicies[setID] = list(np.where(y_test == setID)[0])
        if len(yIndicies[setID]) == 0: continue
        ds_feats = l_feat_te[setID]
        testIndex = yIndicies[setID][0]
        X_test[testIndex] = ds_feats[idx]
        X_idx[testIndex] = {"idx":l_idx_te[setID][idx],"split":"test"}
        yIndicies[setID].remove(testIndex)
    
def roidbToSVMData(roidbTr,roidbTe,train_size,test_size,pyloader=roidbSampleHOG,calcHog=False):
    l_feat_tr,l_idx_tr,y_tr = roidbToFeatures(roidbTr,pyloader=pyloader,calcHog=calcHog)
    X_train, X_test, y_train, y_test, X_idx = split_data(train_size, test_size, \
                                                         l_feat_tr,l_idx_tr, y_tr,\
                                                         clsToSet)
    l_feat_te,l_idx_te,y_te = roidbToFeatures(roidbTe)
    # this is a work-around for the loading of a "testing" mixed dataset... overwrites the original split from the training data

    # mangleTestingData(l_feat_te,l_idx_te,y_te,X_test,y_test,X_idx,test_size)
    X_train, X_test = scale_data(X_train, X_test)
    print(X_train.shape)
    print(y_train.shape)
    return X_train, X_test, y_train, y_test, X_idx
        
if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    cfg.DEBUG = False

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        np.random.seed(cfg.RNG_SEED)

    setID = args.setID
    repeat = args.repeat
    size = args.size
    

    mixedData = load_mixture_set(setID,repeat,size)
    roidbTr,annoCountTr,roidbTe,annoCountTe = mixedData["train"][0],mixedData["train"][1],mixedData["test"][0],mixedData["test"][1]

    # cropped hog image input
    appendHOGtoRoidb(roidbTr)
    appendHOGtoRoidb(roidbTe)

    print("annoCountTr: {}".format(annoCountTr))
    print("annoCountTe: {}".format(annoCountTe))
    print_report(roidbTr,annoCountTr,roidbTe,annoCountTe,setID,repeat,size)

    print("-="*50)

    clsToSet = loadDatasetIndexDict()

    print("as pytorch friendly ")

    train_size = 500
    test_size = 500

    X_train, X_test, y_train, y_test, X_idx = roidbToSVMData(roidbTr,roidbTe,\
                                                             train_size,test_size)
    if args.modelCropped is not None:
        model = pickle.load(open(args.modelCropped,"rb"))
    else:
        model = train_SVM(X_train,y_train)
        pickle.dump(model,open(iconicImagesFileFormat().format("modelCropped_{}_{}_{}.pkl".format(setID,repeat,size)),"wb"))

    print("accuracy on test data {}".format(model.score(X_test,y_test)))

    path_to_save = osp.join(cfg.PATH_TO_NTD_OUTPUT, 'Mat1_'+setID+'_'+repeat+'_'+str(size))
    
    cm_cropped = make_confusion_matrix(model, X_test, y_test, clsToSet)
    plot_confusion_matrix(np.copy(cm_cropped), clsToSet,
                          path_to_save, title="Cropped Images",
                          cmap = plt.cm.bwr_r,vmin=-100,vmax=100,)


    #raw image input 
    X_train, X_test, y_train, y_test, X_idx = roidbToSVMData(roidbTr,roidbTe,\
                                                             train_size,test_size,
                                                             pyloader=roidbSampleImage,
                                                             calcHog=True)
    if args.modelRaw is not None:
        model = pickle.load(open(args.modelRaw,"rb"))
    else:
        model = train_SVM(X_train,y_train)
        pickle.dump(model,open(iconicImagesFileFormat().format("modelRaw_{}_{}_{}.pkl".format(setID,repeat,size)),"wb"))

    print("accuracy on test data {}".format(model.score(X_test,y_test)))
    
    path_to_save = osp.join(cfg.PATH_TO_NTD_OUTPUT, 'Mat2_'+setID+'_'+repeat+'_'+str(size))

    cm_raw = make_confusion_matrix(model, X_test, y_test, clsToSet)
    plot_confusion_matrix(np.copy(cm_raw), clsToSet,
                          path_to_save, title="Raw Images",
                          cmap = plt.cm.bwr_r,vmin=-100,vmax=100,)

    #diff between to cm's
    print("at this line")   
    print(cm_raw)
    print(cm_cropped)
    diff = cm_raw - cm_cropped

    fid = open(iconicImagesFileFormat().format("confMats_{}_{}_{}.pkl".format(setID,repeat,size)),"wb")
    pickle.dump({"raw":cm_raw,"cropped":cm_cropped},fid)
    fid.close()

    path_to_save = osp.join(cfg.PATH_TO_NTD_OUTPUT, 'diff_Mat1_'+setID+'_'+repeat+'_'+str(size)+'Mat2_'+setID+'_'+repeat+'_'+str(size))

    # plotLimit = np.max(np.abs(diff))
    plot_confusion_matrix(diff, 
                          clsToSet, path_to_save, 
                          cmap = plt.cm.bwr_r,
                          show_plot = False,vmin=-100,vmax=100,
                          title="Raw - Cropped")
   
