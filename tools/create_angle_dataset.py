#!/usr/bin/env python
# --------------------------------------------------------
# --------------------------------------------------------

"""Test an object detection network on an image database."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _init_paths
from core.test_new import test_net
from utils.base import writeNdarray
from core.config import cfg, cfg_from_file, cfg_from_list, set_global_cfg, getTestNetConfig, set_dataset_augmentation_config, set_class_inclusion_list_by_calling_dataset,setModelInfo,check_config_for_error,saveExperimentConfig,computeUpdatedConfigInformation,get_output_dir
from utils.create_angle_dataset_utils import optimal_angles_for_net_and_imdb,plot_input_angles_versus_optimal_angles,plot_optimal_angles_histogram
from datasets.factory import get_repo_imdb
from cache.test_results_cache import TestResultsCache
import numpy as np
import caffe
import argparse
import pprint
import time, os, sys


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test an Object Detection network')
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
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_dets_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--rotate', dest='rotate',
                        help='how much should we rotate each image?',
                        default=0, type=int)
    parser.add_argument('--av_save', dest='av_save',
                        help="tells us to save the activity vectors",
                        action='store_true')
    # params for model to which active learning is applied
    parser.add_argument('--al_def', dest='al_def',
                        help='model prototxt to which active learning is applied',
                        default=None, type=str)
    parser.add_argument('--al_net', dest='al_net',
                        help='model weights to which active learning is applied',
                        default=None, type=str)
    parser.add_argument('--warp_affine_pretrain', dest='warp_affine_pretrain',
                        help='did we train the warp affine with a pretrained model?',
                        default=None, type=str)
    parser.add_argument('--name_override', dest='name_override',
                        help='overwrite the current model name with this string instead.',
                        default=None, type=str)
    parser.add_argument('--new_cache', dest='new_cache',
                        help="tells us to re-write the old cache",
                        action='store_true')
    parser.add_argument('--siamese', dest='siamese',
                        help="is the testing model type a siamese model?",
                        action='store_true')
    parser.add_argument('--arch', dest='arch',type=str,
                        help="specify the model architecture")
    parser.add_argument('--optim', dest='optim',type=str,
                        help="specify the model optim alg")
    parser.add_argument('--export_cfg', dest='export_cfg',action='store_true',
                        help="export the config to file.")
    parser.add_argument('--append_save_string', dest='append_save_string',default=None,type=str,
                        help="insert a string at end of dataset name")
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
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    set_global_cfg("TEST")

    cfg.GPU_ID = args.gpu_id
    if args.rotate != 0: cfg.IMAGE_ROTATE = args.rotate
    if args.av_save is False: cfg.SAVE_ACTIVITY_VECTOR_BLOBS = []
    if args.warp_affine_pretrain is True:
        cfg.WARP_AFFINE_PRETRAIN = True
        

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    getTestNetConfig(args.caffemodel,args.prototxt)
    cfg.BATCH_SIZE = 1
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    if args.name_override:
        cfg.modelInfo.name = args.name_override
    else:
        cfg.modelInfo.name = net.name

    # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    # for name,layer in net.layer_dict.items():
    #     if len(layer.blobs) == 0: continue
    #     print(layer.blobs[0].data.shape)
    #     #print(name,layer.blobs[0].data)
    # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    # sys.exit()
    print(args.imdb_name)
    imdb = get_repo_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.OBJ_DET.HAS_RPN and cfg.TASK == 'object_detection':
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    set_dataset_augmentation_config() # reset dataset augmentation settings
    imdb.update_dataset_augmentation(cfg.DATASET_AUGMENTATION)
    set_class_inclusion_list_by_calling_dataset() # reset class inclusion list settings

    al_net = None
    if args.al_net is not None and args.al_def is not None:
        al_net = caffe.Net(args.al_def,caffe.TEST, weights=args.al_net)
        al_net.name = "al_"+os.path.splitext(os.path.basename(args.al_def))[0]

    solverInfo = {'arch':'arch','optim':'optim','siamese':args.siamese}
    if args.arch:
        solverInfo['arch'] = args.arch
    if args.optim:
        solverInfo['optim'] = args.optim

    setModelInfo(None,solverInfo=solverInfo)
    check_config_for_error()
    net_layer_names = [name for name in net._layer_names]
    if 'warp_angle' in net_layer_names and cfg.TASK != "regression":
        cfg.ACTIVATIONS.SAVE_BOOL = True
        cfg.ACTIVATIONS.LAYER_NAMES = ['warp_angle']
    computeUpdatedConfigInformation()
    if args.export_cfg:
        saveExperimentConfig()

    print("Creating an angle dataset for the dataset provided. Augmentation according to the configs.")
    correctness_records = [] # same as in testing currently (01/30/19)
    data_loader = imdb.create_data_loader(cfg,correctness_records,al_net)
    iterations = 300
    output_dir = get_output_dir(imdb, net)
    save_cache = TestResultsCache(output_dir,cfg,imdb.config,None,'angle_dataset')
    data = save_cache.load()
    if data is None or args.new_cache is True:
        input_angles,optimal_angles = optimal_angles_for_net_and_imdb(net,data_loader,iterations,cfg.DATASET_AUGMENTATION.CONFIGS)
        data = [input_angles,optimal_angles]
        save_cache.save(data)
    else:
        input_angles,optimal_angles = data

    print(input_angles)
    plot_input_angles_versus_optimal_angles(input_angles,optimal_angles,scale_angles=False,postfix_string=args.append_save_string)
    plot_optimal_angles_histogram(optimal_angles,scale_angles=False,postfix_string=args.append_save_string)

    
    angle_data = np.transpose(np.vstack((input_angles,optimal_angles)))
    print(angle_data.shape)
    if args.append_save_string:
        savename = "{}_{}".format(cfg.modelInfo.name,args.append_save_string)
    else:
        savename = "{}".format(cfg.modelInfo.name)

    writeNdarray(savename+".npy",angle_data)
    
    print("done.")

    
