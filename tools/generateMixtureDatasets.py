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
from datasets.ds_utils import print_each_size,printPyroidbSetCounts,roidbSampleBox,roidbSampleImageAndBox,combine_roidb,combineOnlyNewRoidbs
from datasets.factory import get_repo_imdb
from ntd.hog_svm import appendHOGtoRoidb
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

#imdb_names = {"coco":1,"pacsal_voc":2,"imagenet":3,"caltech":4,"cam2":5,"inria":6,"sun":7,"kitti":8}

# baby sized sets for debug
train_imdb_names = ["coco-minival2014-default","pascal_voc-medium-default","imagenet-very_short_train-default","cam2-train-default","caltech-train_50_filter-default","kitti-train-default","sun-all-default","inria-all-default"]

# actual sets
#train_imdb_names = ["coco-train2014-default","imagenet-train2014-default","pascal_voc-trainval-default","caltech-train-default","inria-all-default","sun-all-default","kitti-train-default","cam2-all-default"]
test_imdb_names = ["coco-val2014-default","imagenet-val1-default","pascal_voc-test-default","caltech-test-default","inria-all-default","sun-all-default","kitti-val-default","cam2-all-default"]
indexToImdbName = cfg.DATASET_NAMES_ORDERED
datasetSizes = cfg.MIXED_DATASET_SIZES
loadedImdbsTr = {}
loadedImdbsTe = {}

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
    parser.add_argument('--appendHog', dest='appendHog',
                        help='resave the loaded mixed dataset with HOG',
                        action='store_true')
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    return args

def shuffle_imdbs():
    for imdb in loadedImdbsTr.values():
        # imdb.shuffle_image_index()
        imdb.shuffle_roidb()
    for imdb in loadedImdbsTe.values():
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

def matchMangledDatasetName(imdb,proposalDatasetNames):
    for proposalDatasetName in proposalDatasetNames:
        name,imageSet,ds_type = proposalDatasetName.split("-")
        if imdb.name == name and imdb.imageSet == imageSet:
            return True
    return False

def getImdbs(setID):
    imdbsTr,imdbsTe = filterImdbsBySetID(setID)
    # remove ONLY FOR DEBUG
    if cfg.DEBUG:
        if len(imdbsTr) == 0: return []
    else:
        assert len(imdbsTr) == setID.count('1')
    return imdbsTr,imdbsTe


def getMixtureRoidb(imdbs,size,pc):
    numImdbs = len(imdbs)
    mixedRoidb = [None for _ in range(8)]
    annoCounts = [None for _ in range(8)]
    for imdb in imdbs:
        # HACKY af
        # todo: provide a more structural solution
        print(imdb.name)
        if imdb.name in ['sun','inria','cam2'] and size == 1000:
            print("HERE getting 2k instead of 1k")
            sizedRoidb,annoCount = imdb.get_roidb_at_size(2000)
        else:
            sizedRoidb,annoCount = imdb.get_roidb_at_size(size)
        print("type(sizedRoidb): {}".format(type(sizedRoidb)))
        mixedRoidb[imdb.config['setID']] = sizedRoidb
        annoCounts[imdb.config['setID']] = annoCount
        print_each_size(sizedRoidb)
    return mixedRoidb,annoCounts
    
def roidbListToDict(roidbs):
    return dict(zip(cfg.DATASET_NAMES_ORDERED,roidb))

def roidbListOnlyNewToDict(roidbs,pc):
    # assumes ordering of roidbs
    newRoidb = {}
    for idx,roidb in enumerate(roidbs):
        if roidb is None: continue
        #print("idx @ {} : original len(roidb): {} onlyNew len(): {} pc[idx]: {}".format(idx,len(roidb),len(roidb[pc[idx]:]),pc[idx]))
        print(type(roidb))
        print(type(roidb[pc[idx]:]))
        newRoidb[cfg.DATASET_NAMES_ORDERED[idx]] = roidb[pc[idx]:]
    return newRoidb

def clearBboxHOGFtsFromRoidb(roidb):
    for sample in roidb:
        sample['hog'] = None

def addHOGtoNewRoidbSamples(roidbs,pc,size):
    onlyNewSamples = combineOnlyNewRoidbs(roidbs,pcTr)
    if size > 1000: clearBboxHOGFtsFromRoidb(onlyNewSamples)
    appendHOGtoRoidb(onlyNewSamples,size)
    return onlyNewSamples

def createMixtureDataset(setID,size,pcTr,pcTe):
    
    imdbsTr,imdbsTe = getImdbs(setID)
    mixedRoidbTr,annoCountsTr = getMixtureRoidb(imdbsTr,size,pcTr)
    mixedRoidbTe,annoCountsTe = getMixtureRoidb(imdbsTe,size,pcTe)
    assert len([0 for roidb in mixedRoidbTr if roidb is not None]) == setID.count('1')
    # no assert for "mixedRoidbTe" since varies from length
    
    # add hog to the samples
    # Some HACKS to shrink the size of these
    if size != 15000:
        addHOGtoNewRoidbSamples(mixedRoidbTr,pcTr,size)
        addHOGtoNewRoidbSamples(mixedRoidbTe,pcTe,size)
        
    # put the list into a dict
    mixedRoidbDictTr = roidbListOnlyNewToDict(mixedRoidbTr,pcTr)
    mixedRoidbDictTe = roidbListOnlyNewToDict(mixedRoidbTe,pcTe)

    pcTr.update(mixedRoidbTr)
    pcTe.update(mixedRoidbTe)

    return {"train":[mixedRoidbDictTr,annoCountsTr],"test":[mixedRoidbDictTe,annoCountsTe]}


def filterImdbsBySetID(setID):
    global indexToImdbName
    global train_imdb_names
    global test_imdb_names

    imdbListTr = []
    imdbListTe = []

    for idx,char in enumerate(setID):
        if char == '1':
            imdb_name = indexToImdbName[idx]
            imdbListTr.append(loadedImdbsTr[imdb_name])
            if train_imdb_names[idx] != test_imdb_names[idx]:
                imdbListTe.append(loadedImdbsTe[imdb_name])
    return imdbListTr,imdbListTe
    
def loadDatasetsToMemory():
    global loadedImdbsTr
    global loadedImdbsTe
    global train_imdb_names
    global test_imdb_names
    for imdb_name in train_imdb_names:
        imdb, roidb = get_roidb(imdb_name)
        loadedImdbsTr[imdb.name] = imdb
    # only load into test if different from train
    for tr_imdb_name,te_imdb_name in zip(train_imdb_names,test_imdb_names):
        if tr_imdb_name == te_imdb_name: continue
        imdb, roidb = get_roidb(te_imdb_name)
        loadedImdbsTe[imdb.name] = imdb

def print_imdb_report():
    for key,imdb in loadedImdbsTr.items():
        print("{}:".format(imdb.name))
        sys.stdout.write('#images: {:>10}\n'.format(len(imdb.roidb)))
        sys.stdout.write('#annos: {:>11}\n'.format(imdb.roidbSize[-1]))
        print("-"*20)

    for key,imdb in loadedImdbsTe.items():
        print("{}:".format(imdb.name))
        sys.stdout.write('#images: {:>10}\n'.format(len(imdb.roidb)))
        sys.stdout.write('#annos: {:>11}\n'.format(imdb.roidbSize[-1]))
        print("-"*20)

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
    print_imdb_report()
    pcTr = PreviousCounts(8,0)
    pcTe = PreviousCounts(8,0)


    for setNum in range(range_start,range_end+1):
        # for each of the "ranges" we want
        # we want to create the training-testing samples
        # onlyDlDatasets = [ '10000000','01000000','00100000','00010000' ]
        # onlyDlDatasets = [ '10000000','01000000','00100000','00010000' ]
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
                pcTr.zero()
                pcTe.zero()
                for size in datasetSizes:
                    # create a file for each dataset size
                    idlist_filename = createFilenameID(setID,str(r),str(size))
                    mixedData = createMixtureDataset(setID,size,pcTr,pcTe)
                    pklName = idlist_filename + ".pkl"
                    print(pklName)
                    #print(mixedData['train'][1],mixedData['test'][1])
                    if osp.exists(pklName) is False:
                        with open(pklName,"wb") as f:
                            pickle.dump(mixedData,f)
                    else:
                        print("{} exists".format(pklName))

