# metaDatsetGen imports
from core.config import cfg, createFilenameID

# misc imports
from utils.base import *
import pprint
pp = pprint.PrettyPrinter(indent=4)
import pickle,cv2,uuid,os,sys
import os.path as osp
import numpy as np
from numpy import transpose as npt

def load_mixture_set_single(setID,repetition,size):
    pklName = createFilenameID(setID,str(repetition),str(size)) + ".pkl"
    # write pickle file of the roidb
    if osp.exists(pklName) is True:
        fid = open(pklName,"rb")
        loaded = pickle.load(fid)
        fid.close()

        trainData = loaded['train']
        print_each_size(trainData[0])
        testData = loaded['test']
        print_each_size(testData[0])
    else:
        raise ValueError("{} does not exists".format(pklName))
    return trainData,testData

    
def load_mixture_set(setID,repetition,final_size):

    roidbTr = {}
    roidbTe = {}
    l_roidbTr = {}
    l_roidbTe = {}
    annoCountsTr = {}
    annoCountsTe = {}

    datasetSizes = cfg.MIXED_DATASET_SIZES
    if final_size not in datasetSizes:
        print("invalid dataset size")
        print("valid option sizes include:")
        print(datasetSizes)
        raise ValueError("size {} is not in cfg.MIXED_DATASET_SIZES".format(final_size))
    sizeIndex = datasetSizes.index(final_size)
    
    for size in datasetSizes[:sizeIndex+1]:
        print(size)
        # create a file for each dataset size
        pklName = createFilenameID(setID,str(repetition),str(size)) + ".pkl"
        # write pickle file of the roidb
        if osp.exists(pklName) is True:
            fid = open(pklName,"rb")
            loaded = pickle.load(fid)
            fid.close()

            train = loaded['train']
            test = loaded['test']
            print(pklName)
            # todo: what depends on the old version?
            # old version: returns only one list of sizes from "final_size"
            annoCountsTr[size] = train[1]
            annoCountsTe[size] = test[1]
            # if size == final_size: # only save the last count
            #     annoCountsTr = train[1]
            #     annoCountsTe = test[1]
            for dsID,roidb in train[0].items():
                print(dsID,type(roidb),len(roidb))
                l_roidb = list(roidb)
                if dsID not in roidbTr.keys():
                    # sometimes read in as tuple
                    # cause unknown
                    roidbTr[dsID] = list(l_roidb)
                    if size < 5000: print("less than 5k @ {}".format(size))
                    if size < 5000: l_roidbTr[dsID] = list(l_roidb)
                else:
                    print("(a)@ {} roidbTr v.s. l_roidbTr: {} v.s. {}".format(dsID,len(roidbTr[dsID]),len(l_roidbTr[dsID])))
                    roidbTr[dsID].extend(l_roidb)
                    print("(b)@ {} roidbTr v.s. l_roidbTr: {} v.s. {}".format(dsID,len(roidbTr[dsID]),len(l_roidbTr[dsID])))
                    if size < 5000: print("less than 5k @ {}".format(size))
                    if size < 5000: l_roidbTr[dsID].extend(l_roidb)
                    print("(c)@ {} roidbTr v.s. l_roidbTr: {} v.s. {}".format(dsID,len(roidbTr[dsID]),len(l_roidbTr[dsID])))

            for dsID,roidb in test[0].items():
                print(dsID,type(roidb),len(roidb))
                l_roidb = list(roidb)
                if dsID not in roidbTe.keys():
                    roidbTe[dsID] = list(l_roidb)
                    if size < 5000: l_roidbTe[dsID] = list(l_roidb)
                else:
                    roidbTe[dsID].extend(l_roidb)
                    if size < 5000: l_roidbTe[dsID].extend(l_roidb)
        else:
            raise ValueError("{} does not exists".format(pklName))
    # KNOWN ISSUE: the annoCounts* ordering is incorrect.
    # see ymlConfig/default_dataset_index.yml v.s. config DATASET_ORDER 
    for ds in roidbTr.keys():
        print("@ {} roidbTr v.s. l_roidbTr: {} v.s. {}".format(ds,len(roidbTr[ds]),len(l_roidbTr[ds])))
    for ds in roidbTe.keys():
        print("@ {} roidbTe v.s. l_roidbTe: {} v.s. {}".format(ds,len(roidbTe[ds]),len(l_roidbTe[ds])))
    return {"train":[roidbTr,annoCountsTr,l_roidbTr],"test":[roidbTe,annoCountsTe,l_roidbTe]}

def save_mixture_set_single(roidb,annoCount,setID,repetition,size):
    pklName = createFilenameID(setID,str(repetition),str(size)) + ".pkl"
    saveInfo = {"allRoidb":roidb,"annoCounts":annoCount}
    with open(pklName,"wb") as f:
        pickle.dump(saveInfo,f)
            
class PreviousCounts():

    def __init__(self,size,initVal):
        self._prevCounts = [initVal for _ in range(size)]

    def __getitem__(self,idx):
        return self._prevCounts[idx]

    def __str__(self):
        return str(self._prevCounts)

    def update(self,roidbs):
        for idx,roidb in enumerate(roidbs):
            if roidb is None: continue
            self._prevCounts[idx] = len(roidb)

    def zero(self):
        self.setAllTo(0)

    def setAllTo(self,val):
        for idx in range(8):
            self._prevCounts[idx] = val


