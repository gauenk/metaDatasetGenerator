#!/usr/bin/env python

# --------------------------------------------------------
# Img2Vec
# Copyright (c) 2018 GTINC
# Licensed under The MIT License [see LICENSE for details]
# Written by Kent Gauen
# --------------------------------------------------------

"""Train an Img2Vec network on a "region of interest" database."""

import matplotlib
matplotlib.use("Agg")
import _init_paths
from core.train import get_training_roidb
from core.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, loadDatasetIndexDict,createPathRepeat
from datasets.factory import get_repo_imdb
from datasets.ds_utils import load_mixture_set,print_each_size,computeTotalAnnosFromAnnoCount,cropImageToAnnoRegion,printPyroidbSetCounts,roidbSampleImage,roidbSampleImageAndBox,save_mixture_set_single,load_mixture_set_single,pyroidbTransform_cropImageToBox,vis_dets
from ntd.hog_svm import appendHOGtoRoidb,train_SVM
import os.path as osp
import datasets.imdb
import argparse
import pprint
import numpy as np
import numpy.random as npr
import sys,os,cv2

# pytorch imports
from datasets.pytorch_roidb_loader import RoidbDataset

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test loading a mixture dataset')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--setID', dest='setID',
                        help='which 8 digit ID to read from',
                        default='11111111', type=str)
    parser.add_argument('--repeat', dest='repeat',
                        help='which repeat to read from',
                        default='1', type=str)
    parser.add_argument('--size', dest='size',
                        help='which size to read from',
                        default=1000, type=int)
    parser.add_argument('--pyroidb_type', dest='pyroidb_type',
                        help='which type of pyroidb to load',
                        default="mixture", type=str)
    parser.add_argument('--appendHog', dest='appendHog',
                        help='resave the loaded mixed dataset with HOG',
                        action='store_true')
    parser.add_argument('--save', dest='save',
                        help='save some samples with bboxes visualized?',
                        action='store_true')
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

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

def resaveWithHog(setID,repeat):
    datasetSizes = cfg.MIXED_DATASET_SIZES
    for size in datasetSizes:
        roidb,annoCount = load_mixture_set_single(setID,repeat,size)
        print_each_size(roidb)
        appendHOGtoRoidb(roidb)
        save_mixture_set_single(roidb,annoCount,setID,repeat,size)

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

    setID = args.setID
    repeat = args.repeat
    size = args.size
    
    roidb,annoCount = load_mixture_set(setID,repeat,size)
    numAnnos = computeTotalAnnosFromAnnoCount(annoCount)

    print("\n\n-=-=-=-=-=-=-=-=-\n\n")
    print("Report:\n\n")
    print("Mixture Dataset: {} {} {}\n\n".format(setID,repeat,size))

    print("number of images: {}".format(len(roidb)))
    print("number of annotations: {}".format(numAnnos))
    print("size of roidb in memory: {}kB".format(len(roidb) * sys.getsizeof(roidb[0])/1024.))
    print("example roidb:")
    for k,v in roidb[10].items():
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

    path = osp.join(prefix_path,"areas.dat")
    np.savetxt(path,areas,fmt='%.18e',delimiter=' ')
    path = osp.join(prefix_path,"widths.dat")
    np.savetxt(path,widths,fmt='%.18e',delimiter=' ')
    path = osp.join(prefix_path,"heights.dat")
    np.savetxt(path,heights,fmt='%.18e',delimiter=' ')
        
    print("-="*50)

    clsToSet = loadDatasetIndexDict()
    
    # add the "HOG" field to pyroidb
    if args.appendHog:
        resaveWithHog(setID,repeat)

    print("as pytorch friendly ")
    
    if args.pyroidb_type == "mixture":
        pyroidb = RoidbDataset(roidb,[0,1,2,3,4,5,6,7],
                               loader=roidbSampleImageAndBox,
                               transform=pyroidbTransform_cropImageToBox)
    elif args.pyroidb_type == "hog":
        pyroidb = RoidbDataset(roidb,[0,1,2,3,4,5,6,7],
                               loader=roidbSampleHOG,
                               transform=None)
    if args.save:
        print("save 30 cropped annos in output folder.")
        saveDir = "./output/mixedDataReport/"
        if not osp.exists(saveDir):
            print("making directory: {}".format(saveDir))
            os.makedirs(saveDir)
           
        print("pyroidb length: {}".format(len(pyroidb)))
        # randIdx = npr.permutation(len(pyroidb))
        randIdx = np.arange(len(pyroidb))
        for i in range(len(pyroidb)): #range(200):

            # find a specific image
            sample,annoCount = pyroidb.getSampleAtIndex(randIdx[i])

            if sample['image'] != "/srv/sdb1/image_team/pascal_voc/VOCdevkit/VOC0712/JPEGImages/2010_002708.jpg": continue

            img, cls = pyroidb[randIdx[i]]
            ds = clsToSet[cls]
            fn = osp.join(saveDir,"{}_{}.jpg".format(randIdx[i],ds))
            cv2.imwrite(fn,img)
            
            # find a specific image
            sample,annoCount = pyroidb.getSampleAtIndex(randIdx[i])
            print(sample['image'])
            if sample['image'] == "/srv/sdb1/image_team/pascal_voc/VOCdevkit/VOC0712/JPEGImages/2010_002708.jpg":
                fn = osp.join(saveDir,"{}_{}_raw.png".format(ds,i))
                im = cv2.imread(sample['image'])
                cls = sample['gt_classes']
                boxes = sample['boxes']
                vis_dets(im,cls,boxes,i,fn=fn)

'''
argparse.ArgumentParser
Input: (description='Test loading a mixture dataset'), Output: parser

np.zeros
Input: ((size)), Output: areas
    
np.zeros
Input: (size), Output: widths

np.zeros
Input: (size), Output: heights

load_mixture_set_single
Input: (setID,repeat,size), Output: roidb,annoCount

get_bbox_info
Input: (roidb,numAnnos), Output: areas, widths, heights

osp.join
Input: (prefix_path,"areas.dat"), Output: path

RoidbDataset
Input:(roidb,[0,1,2,3,4,5,6,7],
                               loader=roidbSampleImageAndBox,
                               transform=pyroidbTransform_cropImageToBox)
Output: pyroidb
'''
                               
