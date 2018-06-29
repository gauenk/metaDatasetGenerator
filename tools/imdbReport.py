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
from anno_analysis.metrics import annotationDensityPlot,plotDensityPlot

# pytorch imports
from datasets.pytorch_roidb_loader import RoidbDataset

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
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--save', dest='save',default=False,
                        help='save some samples with bboxes visualized?',
                        action='store_true')
    parser.add_argument('--createAnnoMap', dest='createAnnoMap',
                        help='save a created annotation map',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def saveCaltechOneFromEach(roidb):
    prefix_path = cfg.IMDB_REPORT_OUTPUT_PATH
    if osp.exists(prefix_path) is False:
        os.makedirs(prefix_path)
    #index = imdb._get_roidb_index_at_size(50)
    index = len(roidb)
    print("saving 30 imdb annotations to output folder...")        
    print(prefix_path)
    setXX = 0
    vXXX = 0
    setXX_s = 'set{:02}'.format(setXX)
    vXXX_s = 'V{:03}'.format(vXXX)

    for i in range(index):
        print(setXX_s,vXXX_s,roidb[i]['image'])

        if setXX_s in roidb[i]['image'] and vXXX_s in roidb[i]['image']:

            boxes = roidb[i]['boxes']
            if len(boxes) == 0: continue
            im = cv2.imread(roidb[i]['image'])
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            cls = roidb[i]['gt_classes']
            fn = osp.join(prefix_path,"{}_{}_{}_{}.png".format(imdb.name,
                                                               setXX_s,vXXX_s,
                                                               i))
            vXXX += 1
            print(fn)
            vis_dets(im,cls,boxes,i,fn=fn)

        if setXX_s not in roidb[i]['image']:
            setXX += 1
            vXXX = 0

        setXX_s = 'set{:02}'.format(setXX)
        vXXX_s = 'V{:03}'.format(vXXX)
    

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

def vis_dets(im, class_names, dets, _idx_, fn=None, thresh=0.5):
    """Draw detected bounding boxes."""
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(len(dets)):
        bbox = dets[i, :4]
        
        
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    if fn is None:
        plt.savefig("img_{}_{}.png".format(_idx_,str(uuid.uuid4())))
    else:
        plt.savefig(fn.format(_idx_,str(uuid.uuid4())))

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

    # HACK
    for idx,sample in enumerate(roidb):
        if idx == 11:
            print(sample)
            pyroidb = RoidbDataset([sample],[0,1,2,3,4,5,6,7],
                                   loader=roidbSampleImageAndBox,
                                   transform=pyroidbTransform_cropImageToBox)
            idx = 0
            for bimg,target in pyroidb:
                fn = "saveForpaper_{}.png".format(idx)
                cv2.imwrite(fn,bimg)                
                idx += 1
            break

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

    print("-="*50)
    print("mixed datasets roidbsize")
    for size in [50,100,500,1000]:
       sizedRoidb,actualSize = imdb.get_roidb_at_size(size)
       print("@ anno-set-size: {} | num images {} | num annos {}".format(size,len(sizedRoidb),actualSize))
    print("-="*50)

    # issue: we are getting zeros area for 5343 of bboxes for pascal_voc_2007

    path = osp.join(prefix_path,"areas.dat")
    np.savetxt(path,areas,fmt='%.18e',delimiter=' ')
    path = osp.join(prefix_path,"widths.dat")
    np.savetxt(path,widths,fmt='%.18e',delimiter=' ')
    path = osp.join(prefix_path,"heights.dat")
    np.savetxt(path,heights,fmt='%.18e',delimiter=' ')

    pyroidb = RoidbDataset(roidb,[0,1,2,3,4,5,6,7],
                           loader=roidbSampleImageAndBox,
                           transform=pyroidbTransform_cropImageToBox)

    # saveCaltechOneFromEach(roidb)
    print(imdb._get_roidb_index_at_size(5000))
    print(imdb.roidbSize[1000])
    if args.save:
        index = imdb._get_roidb_index_at_size(50)
        print("saving 30 imdb annotations to output folder...")        
        print(prefix_path)
        for i in range(index):
            boxes = roidb[i]['boxes']
            if len(boxes) == 0: continue
            im = cv2.imread(roidb[i]['image'])
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            cls = roidb[i]['gt_classes']
            fn = osp.join(prefix_path,"{}_{}.png".format(imdb.name,
                                                               i))
            print(fn)
            vis_dets(im,cls,boxes,i,fn=fn)

    if args.createAnnoMap:
        clsToSet = loadDatasetIndexDict()
        pyroidb = RoidbDataset(roidb,[0,1,2,3,4,5,6,7],
                               loader=roidbSampleBox,
                               transform=pyroidbTransform_normalizeBox)
        annoMaps = annotationDensityPlot(pyroidb)
        annoMap = annoMaps[clsToSet.index(imdb.name)]
        annoMap /= annoMap.max()
        annoMap *= 255
        print("annoMap: max value: {}".format(annoMap.max()))
        print("annoMap: min value: {}".format(annoMap.min()))
        saveFilename = osp.join(cfg.PATH_TO_ANNO_ANALYSIS_OUTPUT,
                                "annoMap_{}.png".format(imdb.name))
        
        cv2.imwrite(saveFilename,annoMap)
        
'''
argparse.ArgumentParser
Input: (description='Generate an Imdb Report'), Output: parser

get_repo_imdb
Input: (imdb_name), Output: imdb

get_training_roidb
Input: (imdb), Output: roidb

np.zeros
Input: (size), Output: areas

np.zeros
Input:(size), Output: widths

np.zeros
Input: (size), Output: heights

plt.subplots
Input: (figsize=(12, 12), Output: fig, ax

get_roidb
Input: (args.imdb_name), Output: imdb, roidb
 
imdb.roidb_num_bboxes_at
Input: (-1), Output: numAnnos

imdb.get_roidb_at_size
Input: (size), Output: sizedRoidb,actualSize

osp.join
Input: (prefix_path,"areas.dat"), Output: path

RoidbDataset
Input: (roidb,[0,1,2,3,4,5,6,7],
                           loader=roidbSampleImageAndBox,
                           transform=pyroidbTransform_cropImageToBox)
Output:pyroidb

imdb._get_roidb_index_at_size
Input: (30), Output: index
'''
