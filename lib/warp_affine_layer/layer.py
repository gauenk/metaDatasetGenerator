"""
   The layer used to use the warpAffine function in cv2.
   WarpAffineLayer implements a Caffe Python layer.
"""

import caffe,cv2
from easydict import EasyDict as edict
from utils.blob import image_list_to_blobs,blob_list_to_images
from datasets.data_utils.dataset_augmentation_utils import getRotationInfo,translateImage
from utils.timer import Timer
from core.config import cfg
from numpy import transpose as npt
import numpy as np
import numpy.random as npr
import yaml

from caffe.proto import caffe_pb2

class WarpAffineLayer(caffe.Layer):
    """ Layer used to give the warpAffine transformation as a layer"""

    def setup(self, bottom, top):
        """Setup the WarpAffineLayer."""

        self.name = "WarpAffineLayer"
        self.set_net_bool = False
        self.train_mode = False
        self.net = None
        self.layer_name = None
        self.angle_max = 90
        self.angle_star = None

        self.search = edict()
        self.search.step_tolerance = 0.1
        self.search.loss_tolerance = 0.1
        self.search.step_number = 20
        self.search.step_size = 10
        self.search.angle_start = -self.angle_max
        self.search.angle_end = self.angle_max

        lp = caffe_pb2.LayerParameter()
        lp.type = "Softmax"
        layer = caffe.create_layer(lp)
        self.softmax_layer = layer
        data = np.zeros((21,2))
        bottom = [caffe.Blob(data.shape)]
        top = [caffe.Blob([])]
        bottom[0].data[...] = data
        layer.SetUp(bottom, top)
        layer.Reshape(bottom, top)


        self.angle = None
        self.batch_size = cfg.BATCH_SIZE
        top[0].reshape(self.batch_size, 3, cfg.TRAIN.MAX_SIZE,cfg.TRAIN.MAX_SIZE)
        print('[WarpAffineLayer] setup done')


    def set_angle(self,angle):
        self.angle = angle

    def set_net(self,og_net,net,layer_name):
        self.set_net_bool = True
        self.og_net = og_net
        self.net = net
        self.layer_name = layer_name
        for index,layer_name in enumerate(self.net._layer_names):
            if layer_name == self.layer_name:
                break
        self.start_index = index
        self.num_layers = len(self.net._layer_names)

    def compute_output_images(self,input_images,angles,cols,rows):
        output_images = []
        for angle,image in zip(angles,input_images):
            rotationMat, scale = getRotationInfo(angle,cols,rows)
            output_image = cv2.warpAffine(image,rotationMat,(cols,rows),scale)
            output_images.append(output_image)
        return output_images

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector.
        bottom[0] -> input image
        bottom[1] -> parameters for warpAffine
        bottom[2] -> input image shape
        """
        # angle,trans_u,trans_d,trans_l,trans_r,crop,flip = bottom[1].data.reshape(2,3)

        #
        # extract relevant variables
        #
        
        # extract angles
        if self.angle is None:
            angle = bottom[0].data
        else:
            angle = self.angle

        # print("[warp_affine_layer.layer.py] angle.shape",angle.shape)
        # print("[warp_affine_layer.layer.py]: is the copy?",not self.set_net_bool)

        # extract images
        cols,rows = (cfg.TRAIN.MAX_SIZE,cfg.TRAIN.MAX_SIZE)
        image_shape = (cfg.BATCH_SIZE,cfg.COLOR_CHANNEL,cfg.TRAIN.MAX_SIZE,cfg.TRAIN.MAX_SIZE)
        input_blobs = bottom[1].data

        #
        # compute the rotated images
        #

        input_images = blob_list_to_images(input_blobs)
        output_images = self.compute_output_images(input_images,angle,cols,rows)
        output_blobs = image_list_to_blobs(output_images)

        # process for backward layer...
        if self.set_net_bool is True and self.train_mode is True:
            self.angle_star = self.findAngleStar()

        top[0].data[...] = output_blobs.astype(np.float32, copy=False)
        self.set_angle(None)
        
    def findAngleStar(self):
        self.refine_angle_search = True
        self.search_step = 10
        losses = []
        angle_values = []
        angle_values = self.getAngleValuesMinibatch(angle_values,losses)
        while self.refine_angle_search:
            losses = self.lossesFromAnglesMinibatch(angle_values)
            angle_values = self.getAngleValuesMinibatch(angle_values,losses)
        losses = np.array(losses)
        angle_values = np.array(angle_values)
        min_loss = np.min(losses,axis=1)
        min_loss_index = np.argmin(losses,axis=1).ravel()
        row_index = np.arange(angle_values.shape[0])
        angle_star = angle_values[row_index,min_loss_index]
        # print("min loss | angle*")
        # print(np.c_[min_loss,angle_star][:10])
        return angle_star

    def checkLosses(self,losses):
        if np.any(losses < 0):
            print("no losses should be less than 0")
            exit()

    def getAngleValuesMinibatch(self,angle_values,losses):
        self.refine_angle_search = True
        if len(losses) == 0:
            range_start = self.search.angle_start
            range_end = self.search.angle_end
            step_size = np.abs(range_end - range_start) / self.search.step_number
            search_angles = np.tile(np.arange(range_start,range_end+1,step_size),self.batch_size).reshape(self.batch_size,-1)
            self.search.step_size /= 10.
            return np.array(search_angles)

        losses = np.array(losses)
        angle_values = np.array(angle_values)
        min_loss = np.min(losses,axis=1)
        min_loss_index = np.argmin(losses,axis=1).ravel()
        row_index = np.arange(angle_values.shape[0])
        angle_props = angle_values[row_index,min_loss_index]

        if np.mean(min_loss) < self.search.loss_tolerance or self.search.step_size < self.search.step_tolerance:
            self.refine_angle_search = False
            return angle_values

        search_angles_for_minibatch = []
        for angle_prop in angle_props:
            range_start = angle_prop - self.search.step_size * self.search.step_number/2.
            range_end = angle_prop + self.search.step_size * self.search.step_number/2.
            step_size = np.abs(range_end - range_start) / self.search.step_number
            search_angle_for_sample = np.arange(range_start,range_end,step_size)
            search_angles_for_minibatch.append(search_angle_for_sample)
        self.search.step_size /= 10.
        return np.array(search_angles_for_minibatch)

    def getAngleValues(self,angle_values,losses):
        self.refine_angle_search = True
        if len(losses) == 0:
            search_angles = npt(np.tile(np.arange(self.search.angle_start,self.search.angle_end+1,self.search.step_number),self.batch_size).reshape(self.batch_size,-1))
            self.search.step_size /= 10.
            return search_angles

        min_loss = np.min(losses)
        min_loss_index = np.argmin(losses)
        angle_prop = angle_values[min_loss_index]

        if min_loss < self.search.loss_tolerance or self.search.step_size < self.search.step_tolerance:
            self.refine_angle_search = False

        range_start = angle_prop - self.search.step_size * self.search.step_number/2.
        range_end = angle_prop + self.search.step_size * self.search.step_number/2.
        search_angles = np.arange(range_start,range_end,self.search.step_number)
        self.search.step_size /= 10.

        return search_angles

    def lossesFromAnglesMinibatch(self,angle_values_list):
        loss_list = []
        assert self.batch_size == len(angle_values_list),"needs to be one angle list for each sample in minibatch"
        self.search.net_batch_size = len(angle_values_list[0])
        self.net.layers[self.start_index].batch_size = len(angle_values_list[0])
        for sample_index,angle_values_for_sample in enumerate(angle_values_list):
            self.setNetworkInputs(sample_index)
            loss_list_for_sample = self.lossesFromAngles(angle_values_for_sample)
            loss_list.append(loss_list_for_sample)
        return loss_list

    def lossesFromAngles(self,angle_values):
        angle_values = np.array(angle_values)
        self.setAngleValue(angle_values)
        gt_loss = self.net._forward( self.start_index, self.num_layers - 1 )
        cls_output_blob = self.net.blobs['cls_score']
        losses = self.lossFromClsOutput(cls_output_blob)
        return losses

    def lossFromClsOutput(self,cls_output_blob):
        layer = self.softmax_layer
        bottom = [caffe.Blob(cls_output_blob.shape)]
        bottom[0].data[...] = cls_output_blob.data
        top = [caffe.Blob([])]
        # layer.SetUp(bottom, top)
        # layer.Reshape(bottom, top)
        layer.Forward(bottom, top)
        softmax_output = top[0].data
        label = self.og_net.blobs['labels'].data[self.batch_index].astype(np.int)
        softmax_output[np.where(softmax_output < np.finfo(float).eps)[0]] = np.finfo(float).eps
        losses = -np.log(softmax_output[:,label]) / softmax_output.shape[0]
        return losses

    def setAngleValue(self,angle):
        self.net.layers[self.start_index].set_angle(angle)

    def setNetworkInputs(self,batch_index):
        self.batch_index = batch_index
        dataBatch = self.og_net.blobs['data'].data
        labelsBatch = self.og_net.blobs['labels'].data
        batch_size = self.search.net_batch_size
        tile_shape = [batch_size] + [1 for _ in dataBatch.shape[1:]]
        data = np.tile(dataBatch[batch_index,:],tile_shape)
        labels = np.tile(labelsBatch[batch_index,:],tile_shape)
        input_blobs = {'data':data,'labels':labels}
        self.net.layers[0].set_data(input_blobs)
        output = self.net.forward()

    def print_report(self,angleMinibatch,lossesMinibatch):
        print("*"*50)
        for angle_sample,losses_sample in zip(angleMinibatch,lossesMinibatch):
            print(angle_sample.shape)
            print(losses_sample.shape)
            print(np.c_[angle_sample[:,np.newaxis],losses_sample])
            print("-"*10)
        print("^"*50)
            
    def backward(self, top, propagate_down, bottom):
        """ 
        (1) Sample different M's to find the best one and (2) pass gradient to angles
        """

        #
        # extract relevant variables
        #

        # extract input angles
        original_angles_tanh = bottom[0].data
        original_top_gradient = top[0].diff
        original_angles = original_angles_tanh * self.angle_max

        #
        # compute the gradients with sampled rotated images
        #

        #self.angle_star = self.findAngleStar()
        angles_gradient = original_angles - self.angle_star[:,np.newaxis]

        # set the diff
        bottom[0].diff[...] = angles_gradient.astype(np.float32, copy=False)

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        top[0].reshape(self.batch_size, 3, cfg.TRAIN.MAX_SIZE,cfg.TRAIN.MAX_SIZE)


