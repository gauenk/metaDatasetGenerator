#!/usr/bin/env python

# --------------------------------------------------------
# --------------------------------------------------------

"""Reval = re-eval. Re-evaluate saved detections."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _init_paths
from core.test import apply_nms
from core.config import cfg
from datasets.factory import get_repo_imdb
import cPickle
import os, sys, argparse
import numpy as np
from utils.misc import getRotationInfo,centerAndScaleBBox
from core.train import get_training_roidb

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Re-evaluate results')
    parser.add_argument('output_dir', nargs=1, help='results directory',
                        type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to re-evaluate',
                        default='voc_2007-test-default', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--nms', dest='apply_nms', help='apply nms',
                        action='store_true')
    parser.add_argument('--gt', dest='against_gt', help='evaluate against gt',
                        action='store_true')
    parser.add_argument('--rot', dest='rot', help='re-eval with some rotation',
                        default=0, type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def from_dets(imdb_name, output_dir, args):
    imdb = get_repo_imdb(imdb_name)
    imdb.competition_mode(args.comp_mode)

    if args.against_gt:
        dets = build_gt_roidb(imdb,args.rot)
    else:
        with open(os.path.join(output_dir, 'detections.pkl'), 'rb') as f:
            dets = cPickle.load(f)

    if args.apply_nms:
        print 'Applying NMS to all detections'
        nms_dets = apply_nms(dets, cfg.TEST.NMS)
    else:
        nms_dets = dets

    print 'Evaluating detections'
    imdb.evaluate_detections(nms_dets, output_dir)

def build_gt_roidb(imdb,rot):
    roidb = imdb.roidb
    roidb = get_training_roidb(imdb) # for width and height
    num_samples = len(roidb)
    num_classes = len(imdb.classes)
    all_boxes = [ [ [] for _ in xrange(num_samples) ] for _ in xrange(num_classes) ] 
    im_rotates_all = dict.fromkeys(imdb.image_index)
    for idx,sample in enumerate(roidb):
        cols,rows = sample['width'],sample['height']
        rotMat, scale = getRotationInfo(args.rot,cols,rows)
        im_rotates_all[imdb.image_index[idx]] = [[rot,cols,rows]]
        """ if there is rotation we modify the "predicted" bounding boxes
        (that normally simply the groundtruth bounding boxes)
        in the following two ways to establish a fair baseline for comparison with a 
        model. 
        1. move the center of the bbox to the rotated center
        2. scale the GT bbox with the rotated image
        These modifications are needed since the model does not have the freedom
        to rotation a bounding box, yet our evaluation method currently counts this against
        the model
        """
        for cls,box in zip(sample['gt_classes'],sample['boxes']):
            fix_box = centerAndScaleBBox(box,rotMat,scale)
            all_boxes[cls][idx].append(np.append(fix_box,1))
        for cls in range(num_classes):
            all_boxes[cls][idx] = np.array(all_boxes[cls][idx])
            
    dets = {}
    dets["all_boxes"] = all_boxes
    dets["im_rotates_all"] = im_rotates_all
    return dets

if __name__ == '__main__':
    args = parse_args()
    cfg.ROTATE_IMAGE = args.rot

    output_dir = os.path.abspath(args.output_dir[0])
    imdb_name = args.imdb_name
    from_dets(imdb_name, output_dir, args)
    
    '''
argparse.ArgumentParser
Input: (description='Re-evaluate results'), Output: parser

get_repo_imdb
Input: (imdb_name), Output: imdb

apply_nms
Input: (dets, cfg.TEST.NMS), Output: nms_dets

os.path.abspath
Input: (args.output_dir[0]), Output: output_dir
'''
