#!/usr/bin/env python

# --------------------------------------------------------
# Img2Vec
# Copyright (c) 2018 GTINC
# Licensed under The MIT License [see LICENSE for details]
# Written by Kent Gauen
# --------------------------------------------------------

"""Train an Img2Vec network on a "region of interest" database."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _init_paths
from core.train import get_training_roidb
from core.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_repo_imdb
import os.path as osp
import datasets.imdb
import argparse
import pprint
import numpy as np
import numpy.random as npr
import sys,os,cv2,uuid


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
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--txtOutput', dest='txtFilename',
                        help='output text filename',
                        default='comp4_det_test_None.txt', type=str)

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

def imageToId(imagePath):
    base = imagePath.split("/")[-1]
    return base.split(".")[0]

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    txtFilename = args.txtFilename
    imdb, roidb = get_roidb(args.imdb_name)
    numAnnos = imdb.roidb_num_bboxes_at(-1)
    
    f = open(txtFilename,"w+")
    for obj in roidb:
        if obj['flipped']: continue
        for idx,box in enumerate(obj['boxes']):
            if imdb.classes[obj['gt_classes'][idx]] == 'person':
                f.write("{} 1.0 {} {} {} {}\n".format(
                    imageToId(obj['image']),box[0],box[1],
                    box[2],box[3]))
    f.close()
    
    '''
 argparse.ArgumentParser:
 Input: (description='Generate an Imdb Report') Output: parser
     
get_repo_imdb:
Input: (imdb_name), Output: imdb

get_training_roidb:
Input: (imdb), Output: roidb

np.zeros:
 Input: (size), Output: areas
 
np.zeros:
Input: (size), Output: widths

np.zeros:
Inputs: (size), Output: heights

get_roidb
Input: (args.imdb_name), Output: imdb, roidb

imdb.roidb_num_bboxes_at
Input: (-1), Output: numAnnos
    
open
Input: (txtFilename,"w+"), Output: f
'''
