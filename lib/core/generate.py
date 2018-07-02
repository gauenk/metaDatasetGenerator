# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from core.config import cfg
import roi_data_layer.roidb as rdl_roidb
import vae_data_layer.roidb as vae_rdl_roidb
from utils.timer import Timer
import numpy as np
import os,sys,cv2

from caffe.proto import caffe_pb2
import google.protobuf as pb2
import google.protobuf.text_format


class GenerateWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, net, output_dir):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        self.net = net
        self.current_sample_count = 0
        self.display = 2

    def save_sample_set(self,imgs):
        for i in range(imgs.shape[0]):
            self.save_sample(imgs[i])

    def save_sample(self,img):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (infix +
                    '_net_{:s}_{:d}'.format(self.net.name,self.current_sample_count) + '.png')
        filename = os.path.join(self.output_dir, filename)
        # save output as image
        cv2.imwrite(filename,img)
        print('Wrote sample to: {:s}'.format(filename))
        
        self.current_sample_count += 1
        return filename

    def generate(self, numberOfSamples):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        print(dir(self.net.layer_dict["sample"]))
        print(dir(self.net.layer_dict["sample"].blobs))

        while self.current_sample_count < numberOfSamples:
            # Make one SGD update
            timer.tic()
            
            blobs_out = self.net.forward()
            BATCH_SIZE = blobs_out["decode1neuron"].shape[0]
            imgs = blobs_out["decode1neuron"].reshape(BATCH_SIZE,30,30,3) * 255
            self.save_sample_set(imgs)
            timer.toc()
            if self.current_sample_count % (10 * self.display) == 0:
                print('speed: {:.3f}s / iter'.format(timer.average_time))

        return model_paths

def generate_from_net(net, output_dir, numberOfSamples=10):
    """Train *any* object detection network."""

    gw = GenerateWrapper(net,output_dir)
    print('Generating...')
    model_paths = gw.generate(numberOfSamples)
    print('done generating')
    return model_paths
