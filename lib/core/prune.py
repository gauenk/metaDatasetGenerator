# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from core.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import numpy.random as npr
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob,blob_list_im,save_blob_list_to_file
from datasets.ds_utils import cropImageToAnnoRegion
from alcls_data_layer.minibatch import get_minibatch as alcls_get_minibatch
import os
import matplotlib.pyplot as plt
# from sklean.cluster import KMeans


cfg._DEBUG.core.prune = False

def prune_net_iterative_step(net,threshold=0.05):
    for name,layer in net.layer_dict.items():
        if len(layer.blobs) == 0: continue
        print(name)
        for idx in range(len(layer.blobs)):
            data = layer.blobs[idx].data
            mask = layer.blobs[idx].mask
            print("before {}".format(mask.sum()))
            mask[...] = np.where(np.abs(data) < threshold,0,1)
            print("after {}".format(mask.sum()))

def prune_net(net, imdb, max_per_image, vis, al_net):
    # step 1: cluster_the_weights
    cluster_the_weights(net)


def cluster_the_weights(net):
    netCentroids = {}
    netIds = {}
    for name,layer in net.layer_dict.items():
        if len(layer.blobs) == 0: continue
        layerCentroids,layerIds = cluster_the_layer(layer.blobs)
        netCentroids[name] = layerCentroids
        netIds[name] = layerIds
    return netCentroids

def cluster_the_layer(layer_blobs):
    blob = layer_blobs[0].data
    nClusters = 13
    kmeans = KMeans(nClusters).fit(blob)
    return kmeans.cluster_centers_,kmeans.labels_

def initialization(k,initParams):
    initType = 'Linear'
    if initType == 'Linear':
        inits = initLinear(k,initParams)
    elif initType == 'DensityBased':
        inits = initDensityBased(k,initParams)
    elif initType == 'Forgy':
        inits = initForgy(k,initParams)
    return inits

def initLinear(k,initParams):
    inits = np.arange(initParams['min'],initParams['max'],k)
    return inits

def initDensityBased(k,initParams):
    raise NotImplementedError

def initForgy(k,initParams):
    raise NotImplementedError




