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
from utils.misc import createNoisyBox,loadActivityVectors
from core.train import get_training_roidb
from core.config import cfg, cfg_from_file, cfg_from_list, get_output_dir,loadDatasetIndexDict
from datasets.factory import get_repo_imdb
from datasets.ds_utils import load_mixture_set,print_each_size,roidbSampleBox,pyroidbTransform_cropImageToBox,pyroidbTransform_normalizeBox,roidbSampleImageAndBox
import os.path as osp
import datasets.imdb
import argparse
import pprint
import numpy as np
import numpy.random as npr
import sys,os,cv2,uuid
from anno_analysis.metrics import annotationDensityPlot,plotDensityPlot,computeAnnoMapListEntropy

# pytorch imports
from datasets.pytorch_roidb_loader import RoidbDataset

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate information about the activity vectors')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--save', dest='save',default=False,
                        help='save some sample images',
                        action='store_true')
    parser.add_argument('--save_noise', dest='save_noise', type=int,
                        default=0,help='save some samples with *noisy* bboxes visualized?')
    parser.add_argument('--save_box', dest='save_box', default=True,
                        action='store_false',help='save some samples with bboxes visualized')
    parser.add_argument('--createAnnoMap', dest='createAnnoMap',
                        help='save a created annotation map',
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

    
    imdb, roidb = get_roidb(args.imdb_name)
    numAnnos = imdb.roidb_num_bboxes_at(-1)

    av = loadActivityVectors()
    print(av.keys())
    print(av['conv2_1'].keys())
    print(av['conv2_1']['000017'].shape)

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


