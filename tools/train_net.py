#!/usr/bin/env python

"""Train an Img2Vec network on a "region of interest" database."""


    
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _init_paths
from core.train import get_training_roidb, train_net
from core.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, set_global_cfg, set_dataset_augmentation_config, set_class_inclusion_list_by_calling_dataset, setModelInfo,computeUpdatedConfigInformation,saveExperimentConfig
from datasets.factory import get_repo_imdb
from datasets.ds_utils import load_mixture_set,print_each_size
from core.net_utils.initialize_net import mangleSolverPrototxt
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
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=None, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--warp_affine_net', dest='warp_affine_net',
                        help='initialize the warp affine model with model weights',
                        default=None, type=str)
    parser.add_argument('--warp_affine_def', dest='warp_affine_def',
                        help='initialize the warp affine model with model definition',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--dataset_type', dest='dataset_type',
                        help='dataset to train on',
                        default='imdb', type=str)
    parser.add_argument('--mixedSet', dest='mixed_name',
                        help='mized dataset to train on',
                        default='coco', type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='imdb dataset to train on',
                        default='pascal_voc-trainval-default', type=str)
    parser.add_argument('--imdb_size', dest='imdb_size',
                        help='imdb dataset size restriction',
                        default=None, type=int)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--solver_state', dest='solver_state',
                        help='initialize with a previous solver state',
                        default=None, type=str)
    parser.add_argument('--cacheStrModifier', dest='cacheStrModifier',
                        help='append a string to the saved data caches',
                        default=None, type=str)
    # params for model to which active learning is applied
    parser.add_argument('--al_def', dest='al_def',
                        help='model prototxt to which active learning is applied',
                        default=None, type=str)
    parser.add_argument('--al_net', dest='al_net',
                        help='model weights to which active learning is applied',
                        default=None, type=str)
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
    # allow for different path to imagesets:
    parser.add_argument('--new_path_to_imageset',dest='new_path_to_imageset',
                        help='redirect the path from which the imdb will find the imagesets in',
                        default=None,type=str)
    parser.add_argument('--export_cfg', dest='export_cfg',action='store_true',
                        help="export the config to file.")
    parser.add_argument('--append_snapshot_string', dest='append_snapshot_string',default=None,type=str,
                        help="insert a string at end of snapshot name")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_roidb(imdb_name,args):
    imdb = get_repo_imdb(imdb_name,args.new_path_to_imageset,args.cacheStrModifier)
    set_class_inclusion_list_by_calling_dataset() # reset class inclusion list settings
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.OBJ_DET.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.OBJ_DET.PROPOSAL_METHOD))
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
    imdb,roidb = get_roidb(imdb_name,args)
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
    
def set_global_variables_from_args(args):
    if args.solver_state == "None": args.solver_state = None
    if args.pretrained_model == "None": args.pretrained_model = None
    if args.warp_affine_def is not None and args.warp_affine_net is not None:
        cfg.WARP_AFFINE.PRETRAIN = True
        cfg.WARP_AFFINE.PRETRAIN_NAME = os.path.basename(args.warp_affine_net)
    else:
        cfg.WARP_AFFINE.PRETRAIN = False

    if args.cfg_file is not None:
        print("SET CFG FILE")
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id
    set_global_cfg("TRAIN")

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

def get_snapshot_postfix_name(config_postfix,cmd_postfix):
    postfix_string = None
    if config_postfix is not None and cmd_postfix is not None:
        print("[train_net] Two postfix options. Must set them manually. Qutting.")
        exit()
    if config_postfix is not None:
        postfix_string = config_postfix
    if cmd_postfix is not None:
        postfix_string = cmd_postfix
    return postfix_string

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    set_global_variables_from_args(args)

    print('Using config:')
    pprint.pprint(cfg)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    if args.dataset_type == "imdb":
        imdb,roidb,output_dir = get_imdb_dataset(args)
    elif args.dataset_type == "mixed":
        roidb,output_dir = get_mixed_dataset(args)
    else:
        print("Uknown dataset type: {}".format(args.dataset_type))
        sys.exit()

    set_dataset_augmentation_config() # reset dataset augmentation settings

    # activation network for active learning
    al_net = None
    if args.al_net is not None and args.al_def is not None:
        al_net = caffe.Net(args.al_def,caffe.TEST, weights=args.al_net)
        al_net.name = "al_"+os.path.splitext(os.path.basename(args.al_def))[0]

    useDatasetName = True
    datasetName = ""
    if useDatasetName:
        datasetName = imdb.name
    
    if args.max_iters is None and cfg.TRAIN.MAX_ITERS is None:
        args.max_iters = 10000000
    elif args.max_iters is None and cfg.TRAIN.MAX_ITERS is not None:
        args.max_iters = cfg.TRAIN.MAX_ITERS
    elif args.max_iters is not None:
        cfg.TRAIN.MAX_ITERS = args.max_iters
        
    setModelInfo(args.solver)
    print '{:d} roidb entries'.format(len(roidb))
    print 'Output will be saved to `{:s}`'.format(output_dir)
    solver_prototxt = args.solver
    postfix_string = get_snapshot_postfix_name(cfg.MODEL_NAME_APPEND_STRING,args.append_snapshot_string)
    newSolverPrototxt = mangleSolverPrototxt(solver_prototxt,output_dir,cfg.modelInfo,recreate_snapshot_name=True,append_string=postfix_string)
    print("og solver",solver_prototxt)
    computeUpdatedConfigInformation()
    if args.export_cfg:
        saveExperimentConfig()
        print("exported cfg file. now quitting.")
        exit()

    train_net(newSolverPrototxt, imdb, output_dir,
              datasetName=datasetName,
              solver_state=args.solver_state,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters,
              al_net=al_net,
              warp_affine_net=args.warp_affine_net,
              warp_affine_def=args.warp_affine_def)
    
