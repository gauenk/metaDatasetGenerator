#!/usr/bin/env python
# -*- coding: utf-8 -*-

import _init_paths
from core.train import get_training_roidb
from core.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, loadDatasetIndexDict,iconicImagesFileFormat
from datasets.factory import get_repo_imdb
from datasets.ds_utils import load_mixture_set,print_each_size,computeTotalAnnosFromAnnoCount,cropImageToAnnoRegion,roidbSampleHOG,roidbSampleImage
import os.path as osp
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys,os,cv2,pickle
# pytorch imports
from datasets.pytorch_roidb_loader import RoidbDataset
from numpy import transpose as npt
from ntd.hog_svm import plot_confusion_matrix, extract_pyroidb_features,appendHOGtoRoidb,split_data, scale_data,train_SVM,findMaxRegions

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Get iconic image paths from a model')
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
                        default=250, type=int)
    parser.add_argument('--save', dest='save',
                        help='save some samples with bboxes visualized?',
                        action='store_true')
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--model', dest='model',
                        help='give the path to a fit model',
                        default=None, type=str)

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

    path = osp.join(prefix_path,"areas.dat")
    np.savetxt(path,areas,fmt='%.18e',delimiter=' ')
    path = osp.join(prefix_path,"widths.dat")
    np.savetxt(path,widths,fmt='%.18e',delimiter=' ')
    path = osp.join(prefix_path,"heights.dat")
    np.savetxt(path,heights,fmt='%.18e',delimiter=' ')

        
    print("-="*50)

    clsToSet = loadDatasetIndexDict()

    print("as pytorch friendly ")

    pyroidb = RoidbDataset(roidb,[0,1,2,3,4,5,6,7],
                           loader=roidbSampleHOG,
                           transform=None)
    
    print('this is the annocount', annoCount)

    l_feat,l_idx,y = extract_pyroidb_features(pyroidb, 'hog', clsToSet,\
                                              spatial_size=(32, 32),hist_bins=32, \
                                              orient=9, pix_per_cell=8, cell_per_block=2, \
                                              hog_channel=0)

    train_size = 500
    test_size = 500


    X_train, X_test, y_train, y_test, X_idx = split_data(train_size, test_size, \
                                                         l_feat,l_idx, y,\
                                                         clsToSet)
    print(X_train.shape)
    print(y_train.shape)

    if args.model is not None:
        model = pickle.load(open(args.model,"rb"))
    else:
        model = train_SVM(X_train,y_train)
        pickle.dump(model,open(iconicImagesFileFormat().format("model_{}_{}_{}.pkl".format(setID,repeat,size)),"wb"))

    # print("accuracy on test data {}".format(model.score(X_test,y_test)))

    """
    -> below is the raw output for x_test; we want the max "k" values 
    from each dataset (along the columns) from ~1000 images of each dataset
    -> a good "k" is 10
    -> print the image paths to a file
    -> use the format given below
    -> TODO: write the "findMaxRegions" function in "hog_svm.py"
    """

    # rawOutputs = np.matmul(model.coef_,npt(X_test)) + model.intercept_[:,np.newaxis]
    rawOutputs = model.decision_function(X_test)

    print(rawOutputs)
    print(rawOutputs.shape)
    
    fileDir = cfg.PATH_TO_NTD_OUTPUT
    if not osp.exists(fileDir):
        os.makedirs(fileDir)
    
    fileName = osp.join(fileDir,\
                        "{}_{}_{}.txt".format(setID,repeat,size))
    topK = 30
    fn = open(fileName,"w")
    maxRegionsStr = findMaxRegions(topK,pyroidb,rawOutputs,y_test,X_idx,clsToSet)
    fn.write(maxRegionsStr)
    fn.close()
    
    
    '''
    argparse.ArgumentParser:
    Input: (description='create the mixture datasets.'), Output: parser

    np.zeros
    Input: (size), Output: areas

    np.zeros
    Input: (size), Output: width

    np.zeros
    Input: (size), Output: heights

    load_mixture_set
    Input: (setID,repeat,size), Output: roidb, annoCount

    computeTotalAnnosFromAnnoCount
     Input: (annoCount), Output: numAnnos

    get_bbox_info
    Input: roidb, numAnnos, Output: areas, widths, heights


    pyroidb = RoidbDataset
    Input: (roidb,[0,1,2,3,4,5,6,7], loader=roidbSampleHOG, transform=None), Output: pyroidb

    extract_pyroidb_features
    Input: (pyroidb, 'hog', clsToSet,\spatial_size=(32, 32),hist_bins=32, \orient=9, pix_per_cell=8, cell_per_block=2, \hog_channel=0)
    Output: l_feat,l_idx,y

    train_SVM
    Input: (X_train,y_train), Output: model

    np.matmul
    Input: (model.coef_,npt(X_test)) + model.intercept_.shape)
    Output: rawOutputs

     osp.join
    Input: (cfg.PATH_TO_NTD_OUTPUT,\
                            "{}_{}_{}.txt".format(setID,repeat,size))
    Output: fileName

    open
    Input: (fileName,"r"),
    Output: fn
    '''

