#!/usr/bin/env python
# --------------------------------------------------------
# --------------------------------------------------------

"""Test an object detection network on an image database."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _init_paths
from core.test import test_net
from core.config import cfg, cfg_from_file, cfg_from_list, set_global_cfg, getTestNetConfig
from datasets.factory import get_repo_imdb
from utils.blob import im_list_to_blob
from fresh_plot import add_legend_to_axis,adjust_plot_limits

import caffe
import argparse
import pprint
import time, os, sys, cv2
from pprint import pprint as pp

import numpy as np
from numpy import random as npr

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
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--rotate', dest='rotate',
                        help='how much should we rotate each image?',
                        default=0, type=int)
    # params for model to which active learning is applied
    parser.add_argument('--al_def', dest='al_def',
                        help='model prototxt to which active learning is applied',
                        default=None, type=str)
    parser.add_argument('--al_net', dest='al_net',
                        help='model weights to which active learning is applied',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def create_noisy_samples(img,number_per_intensity,noise_intensity,imageID):
    print(noise_intensity)
    noisy_samples = [ [ None for _ in range(number_per_intensity) ] \
                      for _ in range(len(noise_intensity)) ]
    for index,intensity in enumerate(noise_intensity):
        for repeat in range(number_per_intensity):
            noise = npr.rand(*img.shape) * 255. * intensity
            noisy_samples[index][repeat] = img + noise
    plot_tile_of_noise_samples(noisy_samples,noise_intensity,imageID)
    return noisy_samples

def plot_tile_of_noise_samples(noisy_samples,noise_intensity,imageID):

    y_sep = 10 # number of pixels between images along y
    number_of_yaxis_images = 3 # number_of_samples_from_each_noise_intensity
    y_height = number_of_yaxis_images * noisy_samples[0][0].shape[0]

    x_sep = 10 # number of pixels between images along x
    number_of_xaxis_images = len(noise_intensity)
    
    x_buffer = x_sep*number_of_xaxis_images
    y_buffer = y_sep*number_of_yaxis_images
    y_shapes = [images[0].shape[0] for images in noisy_samples] # assume equal sizes
    x_shapes = [images[0].shape[1] for images in noisy_samples]
    tile_shape = [y_height+y_buffer,sum(x_shapes)+x_buffer,3]
    tile = np.zeros(tile_shape)
    x_index_start,x_index_end = 0,0
    for x_index,intensity in enumerate(noise_intensity):
        if x_index != 0: x_index_start += x_shapes[x_index-1] + x_sep
        x_index_end = x_index_start + x_shapes[x_index]
        y_index_start,y_index_end = 0,0
        for y_index,image in enumerate(noisy_samples[x_index][:number_of_yaxis_images]):
            if y_index != 0: y_index_start += y_shapes[y_index-1] + y_sep
            y_index_end = y_index_start + y_shapes[y_index]
            tile[y_index_start:y_index_end,x_index_start:x_index_end,:] = image
    cv2.imwrite("noisy_tile_{}.png".format(imageID),tile)

def plot_class_tallies(class_tallies,noise_intensity,imageID):
    num_classes = len(class_tallies)
    print(num_classes)
    fig,ax = plt.subplots()
    marker_list = ['o','v','^','<','>','8','s','p','h','H','+','x','X','D','d']
    for cls_name,count_along_intensity in class_tallies.items():
        cls_index = imdb.classes.index(cls_name)
        color_value = (cls_index+1)/(1.*num_classes)
        color = plt.cm.RdYlBu(color_value)
        marker = marker_list[cls_index]
        ax.plot(noise_intensity,count_along_intensity,color=color,label=cls_name,marker=marker)
    lgd_title = 'classname'
    ax.set_xlabel('noise intensity')
    ax.set_ylabel('freq')
    ax.set_title('Noisy Classification: {}'.format(imageID))
    adjust_plot_limits(ax,.1,.1)
    add_legend_to_axis([fig,ax],None,None,lgd_title)
    fig.savefig('noisy_classification_{}.png'.format(imageID),bbox_inches='tight')

def test_noisy(net,imdb):
    classes = imdb.classes
    for index in range(20):
        class_name = classes[imdb.roidb[index]['gt_classes'][0]]
        path = imdb.image_path_at(index)
        imageID = imdb.image_index[index]
        test_noisy_image(net,path,classes,class_name,imageID)

def test_noisy_image(net,path,classes,class_name,imageID):
    noise_intensity = np.arange(0.1,1.1,.1)
    number_per_intensity = 100
    print("\n\n\nclassname [{}] at [{}]\n\n\n".format(class_name,path))
    class_tallies = {cls:[0  for _ in noise_intensity] for cls in classes}
    img = cv2.imread(path)
    noisy_imgs = create_noisy_samples(img,number_per_intensity,noise_intensity,imageID)
    for intensity_index,noisy_imgs_at_intensity in enumerate(noisy_imgs):
        noisy_blobs = im_list_to_blob(noisy_imgs_at_intensity)[:,np.newaxis]
        intensity = noise_intensity[intensity_index]
        for index,noisy_blob in enumerate(noisy_blobs):
            class_index = np.argmax(net_forward_blob(net,noisy_blob)['cls_prob'])
            class_name = classes[class_index]
            #print("intensity: {:.2f} | class index: {}".format(intensity,class_index))
            class_tallies[class_name][intensity_index] += 1
    pp(class_tallies)
    plot_class_tallies(class_tallies,noise_intensity,imageID)
    
def net_forward_blob(net,blob):
    output = net.forward(data=blob)
    return output

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
    cfg.DATASET_AUGMENTATION.IMAGE_ROTATE = args.rotate

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    getTestNetConfig(args.caffemodel,args.prototxt)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    
    print(args.imdb_name)
    imdb = get_repo_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.OBJ_DET.HAS_RPN and cfg.TASK == 'object_detection':
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    test_noisy(net, imdb)
