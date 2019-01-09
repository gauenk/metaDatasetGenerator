#!/usr/bin/env python
# --------------------------------------------------------
# --------------------------------------------------------

"""Test an object detection network on an image database."""

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _init_paths
from core.test import test_net
from core.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_repo_imdb
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np


def dtft_of_vgg16(net):
    for name,layer in net.layer_dict.items():
        if name in ["input"]: continue
        print(len(layer.blobs))
        if len(layer.blobs) == 0:
            print(name,"0")
            continue
        weights = layer.blobs[0].data
        print(name,weights.shape)
        plot_weights_v2(weights,name)

def plot_weights_v1(weights,name):
    n_imgs = weights.shape[0]
    n_cols = np.ceil(np.sqrt(n_imgs))
    n_rows = np.ceil(n_imgs/float(n_cols))
    for idx in range(1,n_imgs+1):
        plt.subplot(n_cols,n_rows,idx)
        if idx == 1: plt.title("Layer {}".format(name))
        plt.axis('off')
        plt.imshow(weights[idx-1,:,:,:],interpolation='none')
    plt.show()
    plt.clf()

def plot_weights_v2(weights,name):
    n_cols = weights.shape[0]
    n_rows = weights.shape[1]
    for idx in range(0,n_cols):
        for jdx in range(0,n_rows):
            fig_index = idx*n_rows+jdx + 1
            plt.subplot(n_rows,n_cols,fig_index)
            if idx == 0 and jdx == 0: plt.title("Layer {}".format(name))
            plt.axis('off')
            plt.imshow(weights[idx,jdx,:,:],cmap='Greys',interpolation='none')
            # if jdx > 0:
            #     print(np.sum(weights[idx,jdx-1,:,:]-weights[idx,jdx,:,:]))
            #     if np.allclose(weights[idx,jdx-1,:,:],weights[idx,jdx,:,:]):
            #         print("equal at ({},{}) and ({},{})".format(idx,jdx-1,idx,jdx))
                
    plt.savefig("vgg16_layer_{}.png".format(name),bbox_inches='tight',dpi=1000,transparent=True,frameon=False)
    #plt.show()
    plt.clf()


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
                        action='store_false')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--rotate', dest='rotate',
                        help='how much should we rotate each image?',
                        default=0, type=int)

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

    cfg.GPU_ID = args.gpu_id
    cfg.DATASET_AUGMENTATION.IMAGE_ROTATE = args.rotate

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    print(args.imdb_name)
    imdb = get_repo_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.OBJ_DET.HAS_RPN and cfg.TASK == 'object_detection':
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)


    dtft_of_vgg16(net)



