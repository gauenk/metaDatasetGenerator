# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

Corg implements a Caffe Python layer.
"""

import caffe
import numpy as np
import yaml

class Corg(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        

        self._norm = np.array(layer_params['norm'])
        if 'ssd' in layer_params.keys(): self._ssd = layer_params['norm']
        else: self._ssd = False
        if 'cls' in layer_params.keys(): self._cls = layer_params['cls']
        else: self._cls = False

        self._prob_indicies = np.array(layer_params['indicies'])
        self._nclasses = layer_params.get('nclasses', len(self._prob_indicies))
        if self._ssd: self._prob_dict = dict(zip(self._prob_indicies,range(len(self._prob_indicies))))

        # make the bbox indicies expand to capture 4 indicies
        #print(self._prob_indicies)
        if self._cls is False:
            self._bbox_indicies = np.empty(len(self._prob_indicies)*4,dtype=np.int)
            for index in range(len(self._prob_indicies)):
                bbox_index = 4*index
                self._bbox_indicies[bbox_index] = 4*self._prob_indicies[index]
                self._bbox_indicies[bbox_index+1] = 4*self._prob_indicies[index]+1
                self._bbox_indicies[bbox_index+2] = 4*self._prob_indicies[index]+2
                self._bbox_indicies[bbox_index+3] = 4*self._prob_indicies[index]+3

        # reshape the top to be the size of the number of output classes
        top[0].reshape(1,len(self._prob_indicies))
        print(len(self._prob_indicies))
        if self._cls is False:
            top[1].reshape(1,len(self._bbox_indicies))
            print(self._bbox_indicies)
            print(len(self._bbox_indicies))
        if len(self._prob_indicies) > self._nclasses:
            raise ValueError("Number of classes must be >= length of the index vector")

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        if self._ssd: self.forward_ssd(bottom,top)
        elif self._cls: self.forward_cls(bottom,top)
        else: self.forward_faster_rcnn(bottom,top)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def forward_cls(self,bottom,top):
        probs = bottom[0].data
        blob_probs = probs[:,self._prob_indicies]
        top[0].reshape(blob_probs.shape[0],self._nclasses)
        top[0].data[0:blob_probs.shape[0],0:blob_probs.shape[1]] = blob_probs

    def forward_ssd(self,bottom,top):
        """
        forward_ssd

        a forward pass for ssd model
    
        pick [0,0,*] because the first 
        dim is 1 (setting), and batch size is 1

        """
        rois = bottom[0].data
        classes = rois[0,0,:,1]
        for idx,cls in enumerate(classes):
            classes[idx] = self._prob_dict[cls]
        rois[0,0,:,1] = classes
        top[0].reshape(*rois.shape)
        top[0].data[:,:,:,:] = rois

    def forward_faster_rcnn(self,bottom,top):

        probs = bottom[0].data
        box_preds = bottom[1].data

        blob_box = box_preds[:,self._bbox_indicies]
        blob_probs = probs[:,self._prob_indicies]

        if self._norm == True:
            blob_probs = blob_probs / np.sum(blob_probs,1)[:,np.newaxis]
        
        top[0].reshape(blob_probs.shape[0],self._nclasses)
        top[0].data[0:blob_probs.shape[0],0:blob_probs.shape[1]] = blob_probs

        top[1].reshape(blob_box.shape[0],self._nclasses*4)
        top[1].data[0:blob_box.shape[0],0:blob_box.shape[1]] = blob_box
