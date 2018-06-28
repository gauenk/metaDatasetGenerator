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


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if not args.randomize:
        np.random.seed(cfg.RNG_SEED)
    print('Using config:')
    pprint.pprint(cfg)

    ntdGameInfo = {}
    ntdGameInfo['trainSize'] = 5
    ntdGameInfo['testSize'] = 5

    cfg.DEBUG = False
    cfg.uuid = str(uuid.uuid4())

    setID_l = args.setID
    repeat_l = args.repeat
    size_l = args.size
    assert len(setID_l) == 1 and len(size_l) == 1,\
        "code only works for one size and size currently"

    numEl = len(setID_l) * len(repeat_l) * len(size_l)
    rawMats = []
    croppedMats = []
    diffMats = []

    for setID in setID_l:
        for repeat in repeat_l:
            for size in size_l:
                ntdGameInfo['setID'] = setID
                ntdGameInfo['size'] = size
                ntdGameInfo['repeat'] = 222 # repeat

                """
                convMat_fn = "output/ntd/confMats_{}_{}_{}.pkl".format(setID,repeat,size)
                convMat = pickle.load(open(convMat_fn,"rb"))
                cmRaw = convMat['raw']
                cmCropped = convMat['cropped']
                """
                cmRaw = np.array([31,19,10,1,21,13,5,0,18,21,21,2,12,20,4,1,23,12,23,1,17,18,5,1,0,0,0,100,0,0,0,0,11,9,8,0,61,7,2,1,15,15,16,1,13,33,4,4,2,0,1,0,3,1,92,0,12,6,6,2,3,4,3,65]).reshape(8,8)
                cmCropped = np.array([20,14,18,9,8,12,8,11,12,20,20,8,9,14,7,11,16,19,19,8,10,13,7,9,8,8,8,49,6,6,8,8,8,10,10,4,42,12,5,8,19,16,15,7,8,17,7,11,12,8,11,18,9,7,26,9,11,13,12,9,14,11,9,21]).reshape(8,8)
                cmDiff = cmRaw - cmCropped
                plotNtdConfMats(cmRaw,cmCropped,cmDiff,ntdGameInfo)
                sys.exit()

                rawMats.append(cmRaw)
                croppedMats.append(cmCropped)
                diffMats.append(cmDiff)

    ntdGameInfo['ave'] = len(rawMats)
    ntdGameInfo['std'] = len(rawMats)

    rawMatsAve = np.array(rawMats).mean(axis=0)
    print(rawMatsAve.shape)
    croppedMatsAve = np.array(croppedMats).mean(axis=0)
    diffMatsAve = np.array(diffMats).mean(axis=0)

    plotNtdConfMats(rawMatsAve,croppedMatsAve,diffMatsAve,ntdGameInfo,"ave")

    rawMatsStd = np.array(rawMats).std(axis=0)
    croppedMatsStd = np.array(croppedMats).std(axis=0)
    diffMatsStd = np.array(diffMats).std(axis=0)

    plotNtdConfMats(rawMatsStd,croppedMatsStd,diffMatsStd,ntdGameInfo,"std")

    print(rawMatsAve.shape)
    print("\n\n -=-=-=- uuid: {} -=-=-=- \n\n".format(cfg.uuid))

    




