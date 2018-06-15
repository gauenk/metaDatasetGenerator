#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import sys,os,cv2,pickle
# pytorch imports
from datasets.pytorch_roidb_loader import RoidbDataset
from numpy import transpose as npt
from ntd.hog_svm import plot_confusion_matrix, extract_pyroidb_features,appendHOGtoRoidb,split_data, scale_data,train_SVM,findMaxRegions

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

    cfg.DEBUG = False
    cfg.uuid = str(uuid.uuid4())
    ntdGameInfo = {}
    ntdGameInfo['trainSize'] = 500
    ntdGameInfo['testSize'] = 500

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
                cmRaw,modelRaw = genConfRaw(args.modelRaw, roidbTr, roidbTe, ntdGameInfo)
                cmCropped,modelCropped = genConfCropped(args.modelCropped, roidbTr, roidbTe, ntdGameInfo)

                

    def iconicImagesOutputFilename(ntdGameInfo):
        fileDir = cfg.PATH_TO_NTD_OUTPUT
        if not osp.exists(fileDir):
            os.makedirs(fileDir)
        filename = osp.join(fileDir,\
                            "{}_{}_{}.txt".format(ntdGameInfo['setID'],
                                                  ntdGameInfo['repeat'],
                                                  ntdGameInfo['size']))
        return filename
        
    def getIconicImages(model,X_test,y_test,X_idx,pyroidb,ntdGameInfo):
        rawOutputs = model.decision_function(X_test)
        filename = iconicImagesOutputFilename
        


        
    # print("accuracy on test data {}".format(model.score(X_test,y_test)))

    """
    -> below is the raw output for x_test; we want the max "k" values 
    from each dataset (along the columns) from ~1000 images of each dataset
    -> a good "k" is 10
    -> print the image paths to a file
    -> use the format given below
    -> TODO: write the "findMaxRegions" function in "hog_svm.py"
    """

    # rawOutputs = np.matmul(model.coef_,npt(X_test)) + model.intercept_[:,np.newaxis]
    rawOutputs = model.decision_function(X_test)

    print(rawOutputs)
    print(rawOutputs.shape)
    
    topK = 30
    fn = open(fileName,"w")
    maxRegionsStr = findMaxRegions(topK,pyroidb,rawOutputs,y_test,X_idx,clsToSet)
    fn.write(maxRegionsStr)
    fn.close()
    
    
    '''
    argparse.ArgumentParser:
    Input: (description='create the mixture datasets.'), Output: parser

    np.zeros
    Input: (size), Output: areas

    np.zeros
    Input: (size), Output: width

    np.zeros
    Input: (size), Output: heights

    load_mixture_set
    Input: (setID,repeat,size), Output: roidb, annoCount

    computeTotalAnnosFromAnnoCount
     Input: (annoCount), Output: numAnnos

    get_bbox_info
    Input: roidb, numAnnos, Output: areas, widths, heights


    pyroidb = RoidbDataset
    Input: (roidb,[0,1,2,3,4,5,6,7], loader=roidbSampleHOG, transform=None), Output: pyroidb

    extract_pyroidb_features
    Input: (pyroidb, 'hog', clsToSet,\spatial_size=(32, 32),hist_bins=32, \orient=9, pix_per_cell=8, cell_per_block=2, \hog_channel=0)
    Output: l_feat,l_idx,y

    train_SVM
    Input: (X_train,y_train), Output: model

    np.matmul
    Input: (model.coef_,npt(X_test)) + model.intercept_.shape)
    Output: rawOutputs

     osp.join
    Input: (cfg.PATH_TO_NTD_OUTPUT,\
                            "{}_{}_{}.txt".format(setID,repeat,size))
    Output: fileName

    open
    Input: (fileName,"r"),
    Output: fn
    '''

