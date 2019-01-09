#!/usr/bin/env python

"""Train an Img2Vec network on a "region of interest" database."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _init_paths
from core.train import get_training_roidb, train_net
from core.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, set_global_cfg, set_augmentation_by_calling_dataset, set_class_inclusion_list_by_calling_dataset
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
    parser = argparse.ArgumentParser(description='Create a GAN Prototxt')
    parser.add_argument('--generator_model', dest='generator_model',
                        help='Generator prototxt',type=str)
    parser.add_argument('--discriminator_model', dest='discriminator_model',
                        help='Discriminator prototxt',type=str)
    parser.add_argument('--output_prototxt', dest='output_prototxt',
                        default='gan.prototxt',help='Discriminator prototxt',type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def open_prototxt(prototxt):
    pass

def prototxt_to_layerList(prototxt):
    content = open_prototxt(prototxt)
    pass

def getGanConcatLayer():
    pass

def clear_prototxt(prototxt):
    pass

def add_layer_to_prototxt(prototxt,layer):
    current_layerList = prototxt_to_layerList(prototxt)
    single_layerList = layer_to_prototxt(layer)
    pass

def add_layerList_to_prototxt(prototxt,layerList):
    for layer in layerList:
        add_layer_to_prototxt(prototxt,layer)

def modify_classifier_data_input(prototxt):
    find_layer(prototxt,'ClsInputData')

def concatenate_prototxt(txtA,txtB,txtC):
    layersA = prototxt_to_layerList(txtA)
    layersB = prototxt_to_layerList(txtB)
    ganConcatLayer = getGanConcatLayer()
    # order matters here... of course
    clear_prototxt(txtC)
    add_layerList_to_prototxt(txtC,layersA)
    add_layer_to_prototxt(txtC,ganConcatLayer)
    add_layerList_to_prototxt(txtC,layersB)
    # modify_classifier_data_input(txtC)

if __name__ == "__main__":
    print("HI")
    args = parse_arg()
    
