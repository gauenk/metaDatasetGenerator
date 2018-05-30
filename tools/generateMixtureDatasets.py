#!/usr/bin/env python

# --------------------------------------------------------
# Img2Vec
# Copyright (c) 2018 CAM2
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Get information about the mixture datasets"""

import _init_paths
from utils.misc import PreviousCounts
from core.train import get_training_roidb, train_net
from core.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, createFilenameID, createPathRepeat, createPathSetID
from datasets.ds_utils import print_each_size,printPyroidbSetCounts
from datasets.factory import get_repo_imdb
import datasets.imdb
import numpy as np
import numpy.random as npr
import argparse
import pprint
import numpy as np
import sys,os,pickle,cv2
import os.path as osp

# pytorch imports
from datasets.pytorch_roidb_loader import RoidbDataset


DEBUG = False
#imdb_names = {"coco":1,"pacsal_voc":2,"imagenet":3,"caltech":4,"cam2":5,"inria":6,"sun":7,"kitti":8}
imdb_names = {"pascal_voc-train-default":2,"caltech-train-default":4,"coco-minival2014-default":1,"cam2-train-default":5,"sun-train-default":7,"kitti-train-default":8,"imagenet-train2014-default":3,"inria-train-default":6}
indexToImdbName = ['coco','pascal_voc','imagenet','cam2','caltech','kitti','sun','inria']
datasetSizes = cfg.MIXED_DATASET_SIZES
loadedRoidbs = {}
loadedImdbs = {}

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

def shuffle_imdbs():
    for imdb in loadedImdbs.values():
        # imdb.shuffle_image_index()
        imdb.shuffle_roidb()

def get_roidb(imdb_name):
    imdb = get_repo_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    imdb.set_proposal_method(cfg.TRAIN.OBJ_DET.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.OBJ_DET.PROPOSAL_METHOD)
    _ = get_training_roidb(imdb)
    imdb.shuffle_roidb()
    return imdb, imdb.roidb

def createListFromId(setNum):
    idList = []
    for num in range(256):
        strNum = "{:08b}".format(num)
        if strNum.count('1') == setNum:
            idList.append(strNum)
    return idList

def createMixtureDataset(setID,size):
    
    imdbs = setID_to_imdbs(setID)

    # remove ONLY FOR DEBUG
    if DEBUG:
        if len(imdbs) == 0: return []
    else:
        assert len(imdbs) == setID.count('1')

    imdbSize = size / len(imdbs)
    
    mixedRoidb = [None for _ in range(8)]
    annoCounts = [None for _ in range(8)]
    for imdb in imdbs:
        print(imdb.name,imdb.classes)
        imdb = loadedImdbs[imdb.name]
        sizedRoidb,annoCount = imdb.get_roidb_at_size(size)
        print_each_size(sizedRoidb)
        mixedRoidb[imdb.config['setID']] = sizedRoidb
        annoCounts[imdb.config['setID']] = annoCount
    return mixedRoidb,annoCounts


def filterForTesting(idx):
    if indexToImdbName[idx] in loadedImdbs.keys():
        return False
    return True

def setID_to_imdbs(setID):
    global indexToImdbName

    imdb_list = []
    for idx,char in enumerate(setID):
        if filterForTesting(idx): continue
        if char == '1':
            # load imdb
            imdb_name = indexToImdbName[idx]
            imdb_list.append(loadedImdbs[imdb_name])
    return imdb_list
    
def loadDatasetsToMemory():
    global loadedImdbs
    global loadedRoidbs
    global imdb_names
    for imdb_name in imdb_names.keys():
        imdb, roidb = get_roidb(imdb_name)
        loadedImdbs[imdb.name] = imdb
        loadedRoidbs[imdb.name] = roidb

def combined_roidb(roidbs):
    # assumes ordering of roidbs
    roidb = []
    for r in roidbs:
        print_each_size(roidb)
        roidb.extend(r)
    return roidb

def combineOnlyNew(roidbs,pc):
    newRoidb = []
    for idx,roidb in enumerate(roidbs):
        newRoidb.extend(roidb[pc[idx]:])
    return newRoidb

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

    range_start = args.datasetRangeStart
    range_end = args.datasetRangeEnd
    repeat = args.repeat

    if repeat == 0:
        print("ERROR: repeat is 0.")
        sys.exit()

    loadDatasetsToMemory()
    pc = PreviousCounts(8,0)

    for setNum in range(range_start,range_end+1):
        # for each of the "ranges" we want
        for setID in createListFromId(setNum):
            # create setID folder
            path_setID = createPathSetID(setID)
            if osp.isdir(path_setID) == False:
                os.makedirs(path_setID)
            for r in range(repeat):
                # create the repeat folder
                path_repeat = createPathRepeat(setID,str(r))
                if osp.isdir(path_repeat) == False:
                    os.makedirs(path_repeat)
                # shuffle imdbs
                shuffle_imdbs()
                # reset previuos counters
                pc.zero()
                for size in datasetSizes:
                    # create a file for each dataset size
                    idlist_filename = createFilenameID(setID,str(r),str(size))
                    repo_roidbs,roidbs_anno_counts = createMixtureDataset(setID,size)
                    assert len(repo_roidbs) == setID.count('1')
                    # write pickle file of the roidb
                    allRoidb = combined_roidb(repo_roidbs)
                    onlyNewRoidb = combineOnlyNew(repo_roidbs,pc)
                    pklName = idlist_filename + ".pkl"
                    print(pklName,roidbs_anno_counts,size)
                    pyroidb = RoidbDataset(allRoidb,[0,1,2,3,4,5,6,7],loader=None,transform=None,returnBox=True)
                    printPyroidbSetCounts(pyroidb)
                    pyroidb = RoidbDataset(onlyNewRoidb,[0,1,2,3,4,5,6,7],loader=None,transform=None,returnBox=True)
                    printPyroidbSetCounts(pyroidb)

                    saveInfo = {"allRoidb":onlyNewRoidb,"annoCounts":roidbs_anno_counts}
                    if osp.exists(pklName) is False:
                        with open(pklName,"wb") as f:
                            pickle.dump(saveInfo,f)
                    else:
                        print("{} exists".format(pklName))
                    pc.update(repo_roidbs)
