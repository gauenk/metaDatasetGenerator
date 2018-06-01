#!/usr/bin/env python

# --------------------------------------------------------
# Img2Vec
# Copyright (c) 2018 GTINC
# Licensed under The MIT License [see LICENSE for details]
# Written by Kent Gauen
# --------------------------------------------------------

"""
Given an imdb object and text output detections,
compute the AP
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _init_paths
from core.train import get_training_roidb
from core.config import cfg, cfgData, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_repo_imdb
from datasets.evaluators.bboxEvaluator import bboxEvaluator
import os.path as osp
import datasets.imdb
import argparse
import pprint
import numpy as np
import numpy.random as npr
import sys,os,cv2,uuid,re


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate an Imdb Report')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='coco-train-default', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--txtDets', dest='txtFilename',
                        help='detections to be evaluated',
                        action=None,required=True)
    parser.add_argument('--save', dest='save',
                        help='save some samples with bboxes visualized?',
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

def get_bbox_info(roidb,size):
    areas = np.zeros((size))
    widths = np.zeros((size))
    heights = np.zeros((size))
    actualSize = 0
    idx = 0
    for image in roidb:
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
    return areas,widths,heights

def getResultsFileFormatFromFilename(txtFilename):
    base = txtFilename.split("/")[-1]
    rstr = "(?P<compID>[^_]+)(_(?P<salt>[^_]+))?_det_(?P<imgSet>[^_]+)_(?P<cls>[^_]+)\.txt"
    mgd = re.match(rstr,base).groupdict()

    if mgd['salt'] is None:
        salt = ""
    else:
        salt = "_"+mgd['salt']

    return mgd['compID'],salt,mgd['imgSet'],mgd['cls']

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

    txtFilename = args.txtFilename
    imdb, roidb = get_roidb(args.imdb_name)
    numAnnos = imdb.roidb_num_bboxes_at(-1)
    print("\n\n-=-=-=-=-=-=-=-=-\n\n")
    print("Report:\n\n")
    print("number of classes: {}".format(imdb.num_classes))
    print("number of images: {}".format(len(roidb)))
    print("number of annotations: {}".format(numAnnos))
    print("size of imdb in memory: {}kB".format(sys.getsizeof(imdb)/1024.))
    print("size of roidb in memory: {}kB".format(len(roidb) * sys.getsizeof(roidb[0])/1024.))
    print("example roidb:")
    for k,v in roidb[0].items():
        print("\t==> {},{}".format(k,type(v)))
        print("\t\t{}".format(v))

    print("computing bbox info...")
    areas, widths, heights = get_bbox_info(roidb,numAnnos)

    print("ave area: {} | std. area: {}".format(np.mean(areas),np.std(areas,dtype=np.float64)))
    print("ave width: {} | std. width: {}".format(np.mean(widths),np.std(widths,dtype=np.float64)))
    print("ave height: {} | std. height: {}".format(np.mean(heights),np.std(heights,dtype=np.float64)))
    prefix_path = cfg.IMDB_REPORT_OUTPUT_PATH
    if osp.exists(prefix_path) is False:
        os.makedirs(prefix_path)

    # issue: we are getting zeros area for 5343 of bboxes for pascal_voc_2007

    path = osp.join(prefix_path,"areas.dat")
    np.savetxt(path,areas,fmt='%.18e',delimiter=' ')
    path = osp.join(prefix_path,"widths.dat")
    np.savetxt(path,widths,fmt='%.18e',delimiter=' ')
    path = osp.join(prefix_path,"heights.dat")
    np.savetxt(path,heights,fmt='%.18e',delimiter=' ')

    _compID,_salt,_imageSet,cls = getResultsFileFormatFromFilename(txtFilename)
    print(_compID,_salt,_imageSet,cls)
    print(imdb._image_index[:10])
    print(imdb.name)
    print(imdb._cachedir)
    bbEval = bboxEvaluator(imdb.name,imdb.classes,_compID,_salt,
                           imdb._cachedir,imdb._imageSetPath,
                           imdb._image_index,cfgData['PATH_TO_ANNOTATIONS'],
                           imdb.load_annotation,onlyCls=cls)
    bbEval._pathResults = '/'.join(txtFilename.split("/")[:-1])+"/"
    bbEval._imageSet = _imageSet
    bbEval._do_python_eval("./output/txtEval/")

'''
argparse.ArgumentParser
Input: (description='Generate an Imdb Report'), Output: parser

get_repo_imdb
input: (imdb_name), output: imdb

get_training_roidb
input: (imdb), output: roidb

areas = np.zeros
input: (size), output: areas

np.zeros
input: (size), output: widths

np.zeros
input: (size), output: heights

get_roidb
input: (args.imdb_name), output: imdb, roidb

imdb.roidb_num_bboxes_at
input: (-1), output: numAnnos

get_bbox_info
input: (roidb,numAnnos), output: areas, widths, heights

osp.join
input: (prefix_path,"areas.dat"), output: path

osp.join
input: (prefix_path,"widths.dat"), output: path

osp.join
input: (prefix_path,"heights.dat"), output: path

getResultsFileFormatFromFilename
input: (txtFilename), output: _compID,_salt,_imageSet,cls
'''
