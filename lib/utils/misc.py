import sys,os,pickle
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from core.config import cfg,iconicImagesFileFormat
from ntd.hog_svm import plot_confusion_matrix,appendHOGtoRoidb,split_data, scale_data,train_SVM,findMaxRegions, make_confusion_matrix
from datasets.ds_utils import computeTotalAnnosFromAnnoCount

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


def printSaveBboxInfo(roidb,numAnnos,splitStr):
    print("computing bbox info [{:s}]...".format(splitStr))
    areas, widths, heights = get_bbox_info(roidb,numAnnos)

    print("[{:s}] ave area: {} | std. area: {}".format(splitStr,np.mean(areas),np.std(areas,dtype=np.float64)))
    print("[{:s}] ave width: {} | std. width: {}".format(splitStr,np.mean(widths),np.std(widths,dtype=np.float64)))
    print("[{:s}] ave height: {} | std. height: {}".format(splitStr,np.mean(heights),np.std(heights,dtype=np.float64)))
    prefix_path = cfg.IMDB_REPORT_OUTPUT_PATH
    if osp.exists(prefix_path) is False:
        os.makedirs(prefix_path)

    path = osp.join(prefix_path,"areas_{}.dat".format(splitStr))
    np.savetxt(path,areas,fmt='%.18e',delimiter=' ')
    path = osp.join(prefix_path,"widths_{}.dat".format(splitStr))
    np.savetxt(path,widths,fmt='%.18e',delimiter=' ')
    path = osp.join(prefix_path,"heights_{}.dat".format(splitStr))
    np.savetxt(path,heights,fmt='%.18e',delimiter=' ')
    

def print_report(roidbTr,annoCountTr,roidbTe,annoCountTe,setID,repeat,size):

    numAnnosTr = computeTotalAnnosFromAnnoCount(annoCountTr)
    numAnnosTe = computeTotalAnnosFromAnnoCount(annoCountTe)

    print("\n\n-=-=-=-=-=-=-=-=-\n\n")
    print("Report:\n\n")
    print("Mixture Dataset: {} {} {}\n\n".format(setID,repeat,size))
    print_set_report(roidbTr,numAnnosTr,"train")
    print_set_report(roidbTe,numAnnosTe,"test")
    print("example [train] roidb:")
    for k,v in roidbTr[10].items():
        print("\t==> {},{}".format(k,type(v)))
        print("\t\t{}".format(v))
    printSaveBboxInfo(roidbTr,numAnnosTr,"train")
    printSaveBboxInfo(roidbTe,numAnnosTe,"test")

    
def print_set_report(roidb,numAnnos,splitStr):
    print("number of images [{}]: {}".format(splitStr,len(roidb)))
    print("number of annotations [{}]: {}".format(splitStr,numAnnos))
    print("size of roidb in memory [{}]: {}kB".format(splitStr,len(roidb) * sys.getsizeof(roidb[0])/1024.))

def get_bbox_info(roidb,size):
    areas = np.zeros((size))
    widths = np.zeros((size))
    heights = np.zeros((size))
    actualSize = 0
    idx = 0
    print(size)
    for image in roidb:
        # skipped *flipped* samples
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
    print("actual: {} | theoretical: {}".format(idx,size))
    return areas,widths,heights



def saveNtdConfMats(cmRaw,cmCropped,ntdGameInfo):
    fid = open(iconicImagesFileFormat().format("confMats_{}_{}_{}_{}.pkl".\
                                               format(ntdGameInfo['setID'],
                                                      ntdGameInfo['repeat'],
                                                      ntdGameInfo['size'],
                                                      cfg.uuid)),"wb")
    pickle.dump({"raw":cmRaw,"cropped":cmCropped},fid)
    fid.close()

def plotNtdConfMats(cmRaw,cmCropped,cmDiff,ntdGameInfo,infix=None):

    if infix in ntdGameInfo.keys():
        appendStr = '{}_{}_{}_{}_{}'.format(ntdGameInfo['setID'],ntdGameInfo['size'],
                                                 cfg.uuid,infix,ntdGameInfo[infix])
    else:
        appendStr = '{}_{}_{}_{}'.format(ntdGameInfo['setID'],ntdGameInfo['repeat'],
                                         ntdGameInfo['size'],cfg.uuid)
                                  
    pathToRaw = osp.join(cfg.PATH_TO_NTD_OUTPUT, 'ntd_raw_{}'.format(appendStr))
    pathToCropped = osp.join(cfg.PATH_TO_NTD_OUTPUT, 'ntd_cropped_{}'.format(appendStr))
    pathToDiff = osp.join(cfg.PATH_TO_NTD_OUTPUT, 'ntd_diff_raw_cropped_{}'.format(appendStr))
    plot_confusion_matrix(np.copy(cmRaw), cfg.clsToSet,
                          pathToRaw, title="Raw Images",
                          cmap = plt.cm.bwr_r,vmin=-100,vmax=100)
    plot_confusion_matrix(np.copy(cmCropped), cfg.clsToSet,
                          pathToCropped, title="Cropped Images",
                          cmap = plt.cm.bwr_r,vmin=-100,vmax=100)
    plot_confusion_matrix(np.copy(cmDiff), cfg.clsToSet, 
                          pathToDiff,title="Raw - Cropped",
                          cmap = plt.cm.bwr_r,vmin=-100,vmax=100)


def printRoidbImageNamesToTextFile(roidb,postfix):
    fid = open("output_{}.txt".format(postfix),"w+")
    for sample in roidb:
        fid.write(sample['image']+"\n")
    fid.close()
        




