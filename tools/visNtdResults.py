#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:32:08 2018

@author: zkapach
"""

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
                        default='11111111', type=str)
    parser.add_argument('--repeat', dest='repeat',
                        help='which repeat to read from',
                        default='1', type=str)
    parser.add_argument('--size', dest='size',
                        help='which size to read from',
                        default=250, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def constructFilenameToLoad(setID,repeat,size):
    return iconicImagesFileFormat().format(
        "{}_{}_{}.txt".format(setID,repeat,size))
    
def constructFilenameToSaveImage(setID,repeat,size,name,score):
    return iconicImagesFileFormat().format(
        "{}_{}_{}_{}_{}.jpg".format(setID,repeat,size,name,score))

def loadNtdFile(fn):
    lines = []
    d  = {}
    with open(fn,"r") as f:
        for line in f.readlines():
            print(line)
            sline = line.strip().split(",")
            if len(sline) != 4: continue
            print(float(sline[3]))
            if sline[0] in d.keys():
                d[sline[0]] += [{"path":sline[1],
                              "box":map(int,sline[2].split("_")),
                              "score":float(sline[3])}]
            else:
                d[sline[0]] = [{"path":sline[1],
                             "box":map(int,sline[2].split("_")),
                             "score":float(sline[3])}]
    return d

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    setID = args.setID
    repeat = args.repeat
    size = args.size
    
    fn = constructFilenameToLoad(setID,repeat,size)
    ntdResults = loadNtdFile(fn)
    pprint.pprint(ntdResults)
    for dataset,iconicSamples in ntdResults.items():
        for iconicSample in iconicSamples:
            img = cv2.imread(iconicSample['path'])
            box = iconicSample['box']
            score = iconicSample['score']
            cimg = cropImageToAnnoRegion(img,box)
            saveFn = constructFilenameToSaveImage(setID,repeat,size,dataset,score)
            cv2.imwrite(saveFn,cimg)


    
    
