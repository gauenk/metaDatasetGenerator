#!/usr/bin/env python

"""
Run the annotation analysis on a mixed dataset
"""

import matplotlib
matplotlib.use('Agg')


# metaDatasetGen imports
import _init_paths
from core.train import get_training_roidb
from core.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, loadDatasetIndexDict
from datasets.factory import get_repo_imdb
from datasets.ds_utils import load_mixture_set,print_each_size,computeTotalAnnosFromAnnoCount,cropImageToAnnoRegion,roidbSampleBox,pyroidbTransform_normalizeBox
from anno_analysis.metrics import annotationDensityPlot,plotDensityPlot,metric_1,saveRawAnnoPlot
from ntd.hog_svm import HOGFromImage

# pytorch imports
from datasets.pytorch_roidb_loader import RoidbDataset

# misc imports
import sys,os,cv2,argparse,pprint
import os.path as osp
import numpy as np
import numpy.random as npr

# misc [anno analysis] imports
import pdb,csv
from scipy import ndimage, misc
import pandas as pd
import itertools

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
                        default=-1, type=int)
    parser.add_argument('--size', dest='size',
                        help='which size to read from',
                        default=-1, type=int)
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

    datasetSizes = cfg.MIXED_DATASET_SIZES
    if args.size != -1:
        datasetSizes = [args.size]

    repeatRange = range(10)
    if args.repeat != -1:
        repeatRange = [args.repeat]

    ## load svm
    for size in datasetSizes:
        for repeat in range(repeatRange):
            roidb,annoCount = load_mixture_set(setID,repeat,size)
            print(annoCount)
            numAnnos = computeTotalAnnosFromAnnoCount(annoCount)
            clsToSet = loadDatasetIndexDict()
            pyroidb = RoidbDataset(roidb,[0,1,2,3,4,5,6,7],
                           loader=roidbSampleBox,
                           transform=pyroidbTransform_normalizeBox)

            annoMaps = annotationDensityPlot(pyroidb)
            for idx,annoMap in enumerate(annoMaps):
                print(" -=-=-=- dataset {} -=-=-=-=- ".format(clsToSet[idx]))
                print("M1: {}".format(metric_1(annoMap,10)))
                fnPrefix = "{}_{}_{}".format(clsToSet[idx],size,repeat)
                plotDensityPlot(annoMap,fnPrefix)
                saveRawAnnoPlot(annoMap,fnPrefix)

                fnPrefix = "{}_{}_{}_hog_a".format(clsToSet[idx],size,repeat)
                hogFts = HOGFromImage(annoMap.astype(np.float32))
                saveHogFeatures(hogFts,fnPrefix)

                fnPrefix = "{}_{}_{}_hog_b".format(clsToSet[idx],size,repeat)
                hogFts = HOGFromImage(annoMap.astype(np.float32),
                                      spatial_size=(64,64),3,16,2)
                saveHogFeatures(hogFts,fnPrefix)

                fnPrefix = "{}_{}_{}_hog_c".format(clsToSet[idx],size,repeat)
                hogFts = HOGFromImage(annoMap.astype(np.float32),
                                      spatial_size=(64,64),9,16,2)
                saveHogFeatures(hogFts,fnPrefix)

                fnPrefix = "{}_{}_{}_hog_d".format(clsToSet[idx],size,repeat)
                hogFts = HOGFromImage(annoMap.astype(np.float32),
                                      spatial_size=(64,64),9,8,2)
                saveHogFeatures(hogFts,fnPrefix)


 '''   
 argparse.ArgumentParser
Input: (argparse.ArgumentParser), Output: parser

load_mixture_set
Input: (setID, repeat, size), Output: roidb,annoCount

computeTotalAnnosFromAnnoCount
Input: (annoCount), Output: numAnnos

RoidbDataset
Input: (roidb,[0,1,2,3,4,5,6,7],loader=roidbSampleBox, transform=pyroidbTransform_normalizeBox), Output: pyroidb

annotationDensityPlot
Input: (pyroidb), Output:  annoMaps

"{}_{}_{}_hog_a".format
Input: (clsToSet[idx],size,repeat), Output: fnPrefix
'''

