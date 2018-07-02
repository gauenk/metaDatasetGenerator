#!/usr/bin/env python

"""Train an Img2Vec network on a "region of interest" database."""

import _init_paths
from core.train import get_training_roidb
from core.generate import generate_from_net
from core.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_repo_imdb
from datasets.ds_utils import load_mixture_set,print_each_size
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np
import sys,os

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate Samples From a VAE')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_false')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')

    # mixed dataset parameters
    parser.add_argument('--setID', dest='setID',
                        help='which 8 digit ID to read from',
                        default='11111111', type=str)
    parser.add_argument('--repeat', dest='repeat',
                        help='which repeat to read from',
                        default='1', type=str)
    parser.add_argument('--size', dest='size',
                        help='which size to read from',
                        default=1000, type=int)

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


def prepare_onlyA_roidb(roidb):
    # see "prepate_roidb" for origianl function when imdb is passed in
    # not easy to do for mixed roidbs, so we are not doing that :)
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    sizes = [PIL.Image.open(samples['image']).size for i in range(len(roidb))]
    for i in range(len(roidb)):
        roidb[i]['width'] = sizes[i][0]
        roidb[i]['height'] = sizes[i][1]
        # # need gt_overlaps as a dense array for argmax
        # gt_overlaps = roidb[i]['gt_overlaps'].toarray() # not needed
        # # max overlap with gt over classes (columns)
        # max_overlaps = gt_overlaps.max(axis=1) # not needed
        # # gt class that had the max overlap
        # max_classes = gt_overlaps.argmax(axis=1)  # not needed
        # roidb[i]['max_classes'] = max_classes  # not needed
        # roidb[i]['max_overlaps'] = max_overlaps  # not needed
        # # sanity checks
        # # max overlap of 0 => class should be zero (background)
        # zero_inds = np.where(max_overlaps == 0)[0]
        # assert all(max_classes[zero_inds] == 0)
        # # max overlap > 0 => class should not be zero (must be a fg class)
        # nonzero_inds = np.where(max_overlaps > 0)[0]
        # assert all(max_classes[nonzero_inds] != 0)
    return roidb

def get_imdb_dataset(args):
    imdb_name = args.imdb_name
    imdb_size = args.imdb_size
    cfg.TRAIN.CLIP_SIZE = imdb_size
    imdb,roidb = get_roidb(imdb_name)
    output_dir = get_output_dir(imdb)
    print("final roidb_size: {}".format(len(roidb)))
    return imdb,roidb,output_dir

def get_mixed_dataset(args):
    setID = args.setID
    repeat = args.repeat
    size = args.size
    ds_name = args.mixed_name

    mixedData = load_mixture_set(setID,repeat,size)
    train,test = mixedData["train"],mixedData["test"]
    roidbTr = train[0][ds_name]
    annoCountTr = train[1]
    print("annotation counts for training sets")
    print(len(roidbTr))
    print(annoCountTr)
    roidbTr = prepare_onlyA_roidb(roidbTr)
    output_dir = get_output_dir("{}_{}_{}".format(ds_name,size,repeat))
    return roidbTr,output_dir
    
if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        print("SET CFG FILE")
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)


    cfg.GPU_ID = args.gpu_id

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]


    # if args.dataset_type == "imdb":
    #     imdb,roidb,output_dir = get_imdb_dataset(args)
    # elif args.dataset_type == "mixed":
    #     roidb,output_dir = get_mixed_dataset(args)
    # else:
    #     print("Uknown dataset type: {}".format(args.dataset_type))
    #     sys.exit()

    """
    for idx,sample in enumerate(roidb):
        if "width" not in sample.keys():
            print("no width @ {}".format(idx))
            print(sample.keys())
            print(sample['flipped'])
        
    """

    
    #sys.exit()
    output_dir = "./output/vae/samples"
    # print '{:d} roidb entries'.format(len(roidb))
    print 'Output will be saved to `{:s}`'.format(output_dir)
    generate_from_net(net, output_dir, numberOfSamples=10)
    
