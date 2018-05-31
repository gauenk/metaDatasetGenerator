#!/usr/bin/env python
# -*- coding: utf-8 -*-

import _init_paths
from core.train import get_training_roidb
from core.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, loadDatasetIndexDict
from datasets.factory import get_repo_imdb
from datasets.ds_utils import load_mixture_set,print_each_size,computeTotalAnnosFromAnnoCount,cropImageToAnnoRegion,roidbSampleHOG,roidbSampleImage
import os.path as osp
import argparse
import pprint
pp = pprint.PrettyPrinter(indent=4)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='File for transforming x-dataset-gen results into a table. ')
    parser.add_argument('--type', dest='tableType',
                        help='which format of table to print',
                        default='txt', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def readXDatasetResults(fn):
    lines = []
    xdata = {}
    with open(fn,"r") as f:
        header = f.readline().strip().split(',')[1:]
        print(header)
        for line in f.readlines():
            l = line.strip().split(',')
            xdata[l[0]] = dict(zip(header,map(float,l[1:])))
    return xdata

def createLatexTable(xdata):
    pass
def createTxtTable(xdata):
    outputFile = osp.join("./output/faster_rcnn/","xDatasetGen.txt")
    apDiff = [ [] for _ in range(len(xdata))]
    for trSet,xResults in xdata.items():
        for teSet,perf in xResults.items():
            if trSet == teSet:
                

if __name__ == '__main__':

    args = parse_args()
    print('Called with args:')
    print(args)

    fn = osp.join("./output/faster_rcnn/","xDatasetGen.txt")
    xdata = readXDatasetResults(fn)
    pp.pprint(xdata)

    if args.tableType == "latex":
        createLatexTable()
    elif args.tableType == "txt":
        createTxtTable()        
