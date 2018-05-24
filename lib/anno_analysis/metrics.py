import pdb,csv,sys
from core.config import loadDatasetIndexDict
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
    matr = np.zeros((9,500,500)).astype(np.float64)
    cls_count = np.zeros((9)).astype(np.int)
    for box,cls in pyroidb:
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        matr[cls-1,(xmin-1):xmax, (ymin-1):ymax] += 1
        cls_count[cls-1] += 1
    print(cls_count)
    print(sum(cls_count))
    for idx,cls in enumerate(cls_count):
        if cls == 0: continue
        print("{}: {}".format(clsToSet[idx],cls))
        matr[idx,...] /= cls
    return matr

def metric_1(matr,k,pyroidb):
    """
    the 1st quartile minus the 3rd quartile
    """
    ## M1 ##
    flat = matr.flatten()
    flat.sort()
    topk = flat[-k:]
    botk = flat[:k]
    M1 = topk - botk
    print("M1: ".format(M1))
    return M1

def metric_2(matr,pyroidb):
    ## M2 ##
    return ndimage.uniform_filter(matr, size = 3, mode = 'constant')		
