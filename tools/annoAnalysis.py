"""
Run the annotation analysis on a mixed dataset
"""

# metaDatasetGen imports
import _init_paths
from core.train import get_training_roidb
from core.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, loadDatasetIndexDict
from datasets.factory import get_repo_imdb
from datasets.ds_utils import load_mixture_set,print_each_size,computeTotalAnnosFromAnnoCount,cropImageToAnnoRegion


# pytorch imports
from datasets.pytorch_roidb_loader import RoidbDataset


# misc imports
import sys,os,cv2,argparse,pprint
import os.path as osp
import numpy as np
import numpy.random as npr

# misc [anno analysis] imports
import pdb,csv
from scipy import ndimage, misc
import pandas as pd
import itertools


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
    parser.add_argument('--repetition', dest='repetition',
                        help='which repetition to read from',
                        default='1', type=str)
    parser.add_argument('--size', dest='size',
                        help='which size to read from',
                        default=1000, type=int)
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
    repetition = args.repetition
    size = args.size


    roidb,annoCount = load_mixture_set(setID,repetition,size)
    numAnnos = computeTotalAnnosFromAnnoCount(annoCount)
    clsToSet = loadDatasetIndexDict()
    pyroidb = RoidbDataset(roidb,[1,2,3,4,5,6,7,8],
                           loader=cv2.imread,
                           transform=cropImageToAnnoRegion,
                           returnBox=True)

    

