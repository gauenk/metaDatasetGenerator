#!/usr/bin/env python

# --------------------------------------------------------
# Img2Vec
# Copyright (c) 2018 CAM2
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Get information about the mixture datasets"""

import _init_paths
from core.train import get_training_roidb, train_net
from core.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_repo_imdb
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np
import sys,os
import os.path as osp

imdb_names = {}
loadedRoidbs = []

def parse_args():
    """
    Parse input arguments

    -> dataset_range: specifies which of the "chooses" we generate
       examples:

       (8 choose 3) => dataset_range_start = dataset_range_end = 3

       (8 choose 4) + (8 choose 5) => dataset_range_start = 4, dataset_range_end = 5
    """
    parser = argparse.ArgumentParser(description='create the mixture datasets.')
    parser.add_argument('--dataset_range_start', dest='datasetRangeStart',
                        help='specify which datasets to choose from: start',
                        default=0, type=int)
    parser.add_argument('--dataset_range_end', dest='datasetRangeEnd',
                        help='specify which datasets to choose from: end',
                        default=0, type=int)
    parser.add_argument('--repeat', dest='repeat',
                        help='the number of times it repeats',
                        default=0, type=int)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='an optional config file',
                        default=None, type=str)
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

def createListFromId(setNum):
    idList = []
    for num in range(256):
        strNum = "{:08b}".format(num)
        if strNum.count('1') == setNum:
            idList.append(strNum)
    return idList

def createMixtureDataset(setID,size):
    
    roidbs = setID_to_roidb(setID)
    assert len(roidbs) == setID.count('1')
    
    mixedRoidb = []
    for roidb in roidbs:
        sizedRoidb = get_roidb_sample_at_size(roidb,size)
        mixedRoidb.append(sizedRoidb)
    return mixedRoidb
    
def get_roidb_at_size(roidb,size):
    pass

def setID_to_roidb(setID):
    roidb_list = []
    for idx,char in enumerate(setID):
        if char == '1':
            # load imdb
            roidb_list.append(loadedImdbs[idx])
    return roidb_list
    
def loadImdbsToMemory():
    
    global loadedRoidbs = []
    for idx in range(len(imdb_names)):
        imdb, roidb = get_roidb(imdb_names[idx])
        loadedRoidbs.append(roidb)

def createPathSetID(setID):
    return osp.join(cfg.PATH_MIXTURE_DATASETS,setID)

def createPathRepeat(setID,r):
    return osp.join(cfg.PATH_MIXTURE_DATASETS,setID,r)
    
def createFilenameID(setID,r,size)
    return osp.join(cfg.PATH_MIXTURE_DATASETS,setID,r,size)


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        np.random.seed(cfg.RNG_SEED)

    range_start = args.datasetRangeStart
    range_end = args.datasetRangeStart
    repeat = args.repeat

    if repeat == 0:
        print("ERROR: repeat is 0.")
        sys.exit()

    loadImdbsToMemory()
                              
    for setNum in range(range_start,range_end):
        # for each of the "ranges" we want
        for setID in createListFromId(setNum):
            # create setID folder
            path_setID = createPathSetID(setID)
            if osp.isdir(path_setID) == False:
                os.makedirs(path_setID)
            for r in range(repeat):
                # create the repeat folder
                path_repeat = createPathRepeat(setID,r)
                if osp.isdir(path_repeat) == False:
                    os.makedirs(path_repeat)
                for size in datasetSizes:
                    # create a file for each dataset size
                    idlist_filename = createFilenameID(setID,r,size)
                    f = open(idlist_filename,"w+")
                    roidb = createMixtureDataset(setID,size)
                    repo_roidbs = roidb_sortByRepo(roidb)
                    for dataset, roidb in repo_roidbs:
                        # write the dataset name
                        f.write("DATASET: {}".format(dataset))
                        for image_id in roidb:
                            # write the image id
                            f.write("{}".format(roidb['image']))
                    f.close()
