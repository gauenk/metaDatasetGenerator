import pdb,csv,sys,os,pickle
import os.path as osp
from core.config import cfg,cfgData,loadDatasetIndexDict
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

def annotationDensityPlot(pyroidb):
    """
    From a pyroidb object, create a concentration plot of
    the annotations.
    """
    clsToSet = loadDatasetIndexDict()
    matr = np.zeros((8,500,500)).astype(np.float64)
    cls_count = np.zeros((8)).astype(np.int)
    for box,cls in pyroidb:
        print(box)
        box = box.copy() * 500
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        # why "-1"?
        # if clsToSet[cls] == "kitti":
        #     print(ymin)
        matr[cls, ymin:ymax, xmin:xmax] += 1
        cls_count[cls] += 1
    print(cls_count)
    print(sum(cls_count))
    for idx,cls in enumerate(cls_count):
        if cls == 0: continue
        print("{}: {}".format(clsToSet[idx],cls))
        matr[idx,...] /= cls
    return matr

def metric_1(matr,k):
    """
    the 1st quartile minus the 3rd quartile
    """
    ## M1 ##
    flat = matr.flatten()
    flat.sort()
    topk = flat[-k]
    botk = flat[k]
    M1 = topk - botk
    return M1

def metric_2(matr,pyroidb):
    ## M2 ##
    return ndimage.uniform_filter(matr, size = 3, mode = 'constant')		


def plotDensityPlot(people_mask,fn_prefix,rescaled=True):
    fig, ax = plt.subplots()
    my_max = np.max(people_mask)
    my_min = np.min(people_mask)
    my_mean = np.mean(people_mask)
    print("min: {0:4.4f} \nmax: {1:4.4f}\nmean: {2:4.4f}".format(my_min,my_max,my_mean))

    if rescaled:
        cax = ax.imshow(people_mask,vmin=0,cmap="coolwarm",interpolation="none")
        fn = '{}_people_density_rescaled.png'.format(fn_prefix)
    else:
        cax = ax.imshow(people_mask,vmin=0,vmax=0.5,cmap="coolwarm",interpolation="none")
        fn = '{}_people_density.png'.format(fn_prefix)
    #cbar = fig.colorbar(cax,ticks=np.ma.log([my_min,my_mean,my_max]))
    cbar = fig.colorbar(cax,ticks=[my_min,my_mean,my_max])
    cbar.ax.set_yticklabels(['< {0:2.2f}%'.format(my_min*100), '< {0:2.2f}%'.format(my_mean*100), '> {0:2.2f}%'.format(my_max*100)],size=50)
    ax.axis("off")

    saveDir = cfg.PATH_TO_ANNO_ANALYSIS_OUTPUT
    if not osp.exists(saveDir):
        os.makedirs(saveDir)
    fn = osp.join(saveDir,fn)
    plt.savefig(fn,bbox_inches='tight')
                                                                                    

def saveRawAnnoPlot(annoMap,fnPrefix):
    saveDir = cfg.PATH_TO_ANNO_ANALYSIS_OUTPUT
    if not osp.exists(saveDir):
        os.makedirs(saveDir)
    fn = "{}_raw_annomap.pkl".format(fnPrefix)
    fn = osp.join(saveDir,fn)
    with open(fn,"wb") as f:
        pickle.dump(annoMap,f)


