#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import sys,os,cv2,pickle,uuid
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
                        default=['11111111'],nargs='*',type=str)
    parser.add_argument('--repeat', dest='repeat',
                        help='which repeat to read from',
                        default=[1],nargs='*',type=int)
    parser.add_argument('--size', dest='size',
                        help='which size to read from',
                        default=[250],nargs='*',type=int)
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

    l_feat,l_idx,y = extract_pyroidb_features(pyroidb, 'hog', cfg.clsToSet, calc_feat = calcHog, \
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
                                                         cfg.clsToSet)
    l_feat_te,l_idx_te,y_te = roidbToFeatures(roidbTe)
    # this is a work-around for the loading of a "testing" mixed dataset... overwrites the original split from the training data

    # mangleTestingData(l_feat_te,l_idx_te,y_te,X_test,y_test,X_idx,test_size)
    X_train, X_test = scale_data(X_train, X_test)
    print(X_train.shape)
    print(y_train.shape)
    return X_train, X_test, y_train, y_test, X_idx
        
def prepareMixedDataset(setID,repeat,size):
    mixedData = load_mixture_set(setID,repeat,size)
    roidbTr,annoCountTr,roidbTe,annoCountTe = mixedData["train"][0],mixedData["train"][1],mixedData["test"][0],mixedData["test"][1]

    # cropped hog image input
    appendHOGtoRoidb(roidbTr)
    appendHOGtoRoidb(roidbTe)

    print("annoCountTr: {}".format(annoCountTr))
    print("annoCountTe: {}".format(annoCountTe))
    print_report(roidbTr,annoCountTr,roidbTe,annoCountTe,setID,repeat,size)

    print("-="*50)

    return roidbTr,roidbTe


def loadModel(modelFn,modelStr,setID,repeat,size,X_train,y_train):
    if modelFn is not None:
        model = pickle.load(open(modelFn,"rb"))
    else:
        model = train_SVM(X_train,y_train)
        pickle.dump(model,open(iconicImagesFileFormat().format("model{}_{}_{}_{}.pkl".format(modelStr,setID,repeat,size)),"wb"))
    print("\n\n-=- model loaded -=-\n\n")
    return model
    
def genConfCropped(modelFn,roidbTr,roidbTe,ntdGameInfo):
    return genConf(modelFn,"Cropped",roidbTr,roidbTe,roidbSampleHOG,False,ntdGameInfo)

def genConfRaw(modelFn,roidbTr,roidbTe,ntdGameInfo):
    return genConf(modelFn,"Raw",roidbTr,roidbTe,roidbSampleImage,True,ntdGameInfo)

def genConf(modelFn,modelStr,roidbTr,roidbTe,pyloader,calcHog,ntdGameInfo):
    X_train, X_test, y_train, y_test, X_idx = roidbToSVMData(roidbTr,roidbTe,\
                                                             ntdGameInfo['trainSize'],
                                                             ntdGameInfo['testSize'],
                                                             pyloader=pyloader,
                                                             calcHog=calcHog)
    model = loadModel(modelFn,modelStr,ntdGameInfo['setID'],ntdGameInfo['repeat'],
                      ntdGameInfo['size'],X_train,y_train)
    print("accuracy on test data {}".format(model.score(X_test,y_test)))
    return make_confusion_matrix(model, X_test, y_test, cfg.clsToSet)

def saveConfMats(cmRaw,cmCropped,ntdGameInfo):
    fid = open(iconicImagesFileFormat().format("confMats_{}_{}_{}_{}.pkl".\
                                               format(ntdGameInfo['setID'],
                                                      ntdGameInfo['repeat'],
                                                      ntdGameInfo['size'],
                                                      cfg.uuid)),"wb")
    pickle.dump({"raw":cmRaw,"cropped":cmCropped},fid)
    fid.close()

def plotConfMats(cmRaw,cmCropped,cmDiff,ntdGameInfo):
    appendStr = '{}_{}_{}_{}'.format(ntdGameInfo['setID'],ntdGameInfo['repeat'],
                                     ntdGameInfo['size'],cfg.uuid)
    pathToRaw = osp.join(cfg.PATH_TO_NTD_OUTPUT, 'ntd_raw_{}'.format(appendStr))
    pathToCropped = osp.join(cfg.PATH_TO_NTD_OUTPUT, 'ntd_cropped_{}'.format(appendStr))
    pathToDiff = osp.join(cfg.PATH_TO_NTD_OUTPUT, 'ntd_diff_raw_cropped_{}'.format(appendStr))
    plot_confusion_matrix(np.copy(cmRaw), cfg.clsToSet,
                          pathToRaw, title="Raw Images",
                          cmap = plt.cm.bwr_r,vmin=-100,vmax=100)
    plot_confusion_matrix(np.copy(cmCropped), cfg.clsToSet,
                          pathToCropped, title="Cropped Images",
                          cmap = plt.cm.bwr_r,vmin=-100,vmax=100)
    plot_confusion_matrix(np.copy(cmDiff), cfg.clsToSet, 
                          pathToDiff,title="Raw - Cropped",
                          cmap = plt.cm.bwr_r,vmin=-100,vmax=100)

    
    
if __name__ == '__main__':
    
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    cfg.DEBUG = False
    cfg.uuid = str(uuid.uuid4())
    ntdGameInfo = {}
    ntdGameInfo['trainSize'] = 5
    ntdGameInfo['testSize'] = 5

    print('Using config:')
    # pprint.pprint(cfg)

    if not args.randomize:
        np.random.seed(cfg.RNG_SEED)

    setID_l = args.setID
    repeat_l = args.repeat
    size_l = args.size

    for setID in setID_l:
        for repeat in repeat_l:
            for size in size_l:
                ntdGameInfo['setID'] = setID
                ntdGameInfo['repeat'] = repeat
                ntdGameInfo['size'] = size

                roidbTr,roidbTe = prepareMixedDataset(setID,repeat,size)
                cmRaw = genConfRaw(args.modelRaw, roidbTr,roidbTe, ntdGameInfo)
                cmCropped = genConfCropped(args.modelCropped, roidbTr, roidbTe, ntdGameInfo)


                cmDiff = cmRaw - cmCropped

                saveConfMats(cmRaw,cmCropped,ntdGameInfo)
                plotConfMats(cmRaw,cmCropped,cmDiff,ntdGameInfo)
                print("\n\n -=-=-=- uuid: {} -=-=-=- \n\n".format(cfg.uuid))

   
