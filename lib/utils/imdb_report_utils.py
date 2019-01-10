import sys,os,pickle,uuid,cv2,glob,csv
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import numpy.random as npr
from core.config import cfg,iconicImagesFileFormat
from core.routingConfig import cfg as cfgRouting
from ntd.hog_svm import plot_confusion_matrix,appendHOGtoRoidb,split_data, scale_data,train_SVM,findMaxRegions, make_confusion_matrix
from datasets.ds_utils import computeTotalAnnosFromAnnoCount

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
    # legacy
    printRoidbReport(roidbTr,annoCountTr,roidbTe,annoCountTe,setID,repeat,size)

def printTrainTestRoidbReport(roidbTr,annoCountTr,roidbTe,annoCountTe,setID,repeat,size):

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

def printTrainTestRoidbDictReport(roidbTr,annoCountTr,roidbTe,annoCountTe,setID,repeat,size):

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

    
def printRoidbReport(roidb,numAnnos,splitStr):
    print("number of images [{}]: {}".format(splitStr,len(roidb)))
    print("number of annotations [{}]: {}".format(splitStr,numAnnos))
    print("size of roidb in memory [{}]: {}kB".format(splitStr,len(roidb) * sys.getsizeof(roidb[0])/1024.))

def print_set_report(roidb,numAnnos,splitStr):
    # legacy
    printRoidbReport(roidb,numAnnos,splitStr)

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

