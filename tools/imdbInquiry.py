#!/usr/bin/env python

# --------------------------------------------------------
# MDG
# Licensed under The MIT License [see LICENSE for details]
# Written by CAM2
# --------------------------------------------------------

"""
A file to ask some basic questions about the available datasets from imdb
"""

import _init_paths
from core.train import get_training_roidb, train_net
from core.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb,list_imdbs
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate Sample Datasets for MDG')
    parser.add_argument('--mixtureKeyFile', dest='mixture_key',
                        help='the output file for the mixture key',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--list_imdbs', dest='list_imdbs',
                        help='if flag given it lists the available imdbs',
                        action='store_true')
    parser.add_argument('--list_imdbs_with_info', dest='list_imdbs',
                        help='if flag given it lists the available imdbs',
                        action='store_true')
    parser.add_argument('--list_imdbs_sets', dest='list_imdbs_sets',
                        help='if flag given it lists the available imdbs and their training sets',
                        action='store_true')
    parser.add_argument('--list_imdbs_paths', dest='list_imdbs_paths',
                        help="if flag given it lists the available imdbs' paths",
                        action='store_true')
    parser.add_argument('--imdb_size', dest='imdb_size',
                        help="if flag given it lists the available imdbs' paths",
                        action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    For 8 datasets using "inclusion/exclusion", we get 2^8 -1  = 255 sample datasets
    Repeat each datasets K times. If [K = 10] then we have 2,550 sample datasets
    Let's pick 2 obj detector models; pick 5 bases; total of 10 models for each dataset
    In total we must train and test 10 * 2,550 = 25,500 models
    We can scale back to 3 bases to get 6 models and then
    In total we must train and test 6 * 2,550 = 15,300 models
    """
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    if args.list_imdbs:
        print(list_imdbs())

    if args.list_imdbs_sets:
        pass

    if args.list_imdbs_paths:
        pass

