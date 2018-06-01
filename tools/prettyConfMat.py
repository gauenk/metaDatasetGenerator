#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:32:08 2018

@author: zkapach
"""
import matplotlib
# matplotlib.use("Agg")


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
    repeat = args.repeat
    size = args.size
    clsToSet = loadDatasetIndexDict()

    
    convMat_fn = "output/ntd/confMats_11111111_1_1000.pkl"
    convMat = pickle.load(open(convMat_fn,"rb"))
    print(convMat)

    path_to_save = osp.join(cfg.PATH_TO_NTD_OUTPUT, 'ntd_cropped_confusion_matrix.png')
    cm_cropped = convMat['cropped']
    plot_confusion_matrix(np.copy(cm_cropped), clsToSet, path_to_save, title="Cropped Images",
                          show_plot=False)

    path_to_save = osp.join(cfg.PATH_TO_NTD_OUTPUT, 'ntd_raw_confusion_matrix.png')
    cm_raw = convMat['raw']
    plot_confusion_matrix(np.copy(cm_raw), clsToSet, path_to_save, title="Raw Images")

    path_to_save = osp.join(cfg.PATH_TO_NTD_OUTPUT, 'ntd_diff_raw_cropped_confusion_matrix.png')
    diff = cm_raw - cm_cropped
    plot_confusion_matrix(np.copy(diff), clsToSet,
                          path_to_save,
                          cmap = plt.cm.bwr_r,
                          title="Raw - Cropped",
                          vmin=-100,vmax=100)


    




