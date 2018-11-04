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
import sys,os,cv2,pickle,uuid,glob,yaml
from easydict import EasyDict as edict
# pytorch imports
from datasets.pytorch_roidb_loader import RoidbDataset
from numpy import transpose as npt
from utils.misc import *
from ntd.ntd_utils import *
from ntd.hog_svm import plot_confusion_matrix, extract_pyroidb_features,appendHOGtoRoidb,split_data, scale_data,train_SVM,findMaxRegions, make_confusion_matrix

# baby sized for debug
train_imdb_names = ["coco-minival2014-default","pascal_voc-medium-default","imagenet-very_short_train-default","cam2-train-default","caltech-train_50_filter-default","kitti-train-default","sun-all-default","inria-all-default"]
#train_imdb_names = ["coco-train2014-default","imagenet-train2014-default","pascal_voc-trainval-default","caltech-train-default","inria-all-default","sun-all-default","kitti-train-default","cam2-all-default"]
test_imdb_names = ["coco-val2014-default","imagenet-val1-default","pascal_voc-test-default","caltech-test-default","inria-all-default","sun-all-default","kitti-val-default","cam2-all-default"]
indexToImdbName = cfg.DATASET_NAMES_ORDERED
datasetSizes = cfg.MIXED_DATASET_SIZES
loadedImdbsTr = {}
loadedImdbsTe = {}


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

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def loadRecordsDictionary():
    records = {}
    pathsOfRecords = "./experiments/cfgs/ntd_TruePositives_FalseNegatives.yml"
    with open(pathsOfRecords, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    for datasetName,pathOfRecord in yaml_cfg.items():
        records[datasetName] = loadRecord(pathOfRecord)
    print(records)
    return records

def loadRecord(path):
    if path is None or not osp.isfile(path):
        return None
    with open(path,'r') as f:
        record = pickle.load(f)
    return record


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

    records = loadRecordsDictionary()
    
    
    cfg.DEBUG = False
    cfg.uuid = str(uuid.uuid4())
    ntdGameInfo = {}
    ntdGameInfo['trainSize'] = 1000
    ntdGameInfo['testSize'] = 1000
    ntdGameInfo['TP'] = {}
    ntdGameInfo['FN'] = {}
    ntdGameInfo['TP']['trainSize'] = 300
    ntdGameInfo['TP']['testSize'] = 300
    ntdGameInfo['FN']['trainSize'] = 300
    ntdGameInfo['FN']['testSize'] = 300

    modelParams = {}
    modelParams['modelType'] = "dl"
    modelParams['dl_arch'] = "vgg16"
    modelParams['modelType'] = "svm"

    setID_l = args.setID
    repeat_l = args.repeat
    size_l = args.size

    cmTP_l = []
    cmFN_l = []
    cmDiff_l = []
    for setID in setID_l:
        for repeat in repeat_l:
            for size in size_l:


                ntdGameInfo['setID'] = setID
                ntdGameInfo['repeat'] = repeat
                ntdGameInfo['size'] = size
                print("REAPEAT {}".format(repeat))

                # no hack
                roidbTrDict,roidbTeDict,roidbTrDict1k,roidbTeDict1k,dsHasTest,annoSizes = prepareMixedDataset(setID,repeat,size)

                # the original meaning of "mixedRoidb" train and test are not valid in this experiment, so they are combined
                print(roidbTrDict.keys())
                print(roidbTrDict['sun'])
                print(roidbTrDict['sun'][0])
                
                #roidb = roidbTrDict,roidbTeDict
                
                ntdGameInfo["dsHasTest"] = dsHasTest
                print(dsHasTest)
                print(annoSizes)


                modelParams['modelFn'] = None
                if len(args.modelTP) > repeat: modelParams['modelFn'] = args.modelTP[repeat]
                roidbTr = flattenRoidbDict(roidbTrDict)
                roidbTe = flattenRoidbDict(roidbTeDict)
                print("roidb length of:\ntrain: {}\ntest: {}\n".format(len(roidbTr),len(roidbTe)))
                cmTP,modelTP = genConfTP(modelParams, roidbTr, roidbTe, ntdGameInfo)

                modelParams['modelFn'] = None
                if len(args.modelFN) > repeat: modelParams['modelFn'] = args.modelFN[repeat]
                roidbTr = flattenRoidbDict(roidbTrDict1k)
                roidbTe = flattenRoidbDict(roidbTeDict1k)
                print("roidb length of:\ntrain: {}\ntest: {}\n".format(len(roidbTr),len(roidbTe)))
                cmFN,modelFN = genConfFN(modelParams, roidbTr, roidbTe, ntdGameInfo)

                cmDiff = cmTP - cmFN
                saveNtdConfMats(cmTP,cmFN,ntdGameInfo)
                plotNtdConfMats(cmTP,cmFN,cmDiff,ntdGameInfo)
                cmTP_l.append(cmTP)
                cmFN_l.append(cmFN)
                cmDiff_l.append(cmDiff)

    saveNtdSummaryStats(cmTP_l,cmFN_l,cmDiff_l)
    print("\n\n -=-=-=- uuid: {} -=-=-=- \n\n".format(cfg.uuid))
   
