
"""The data layer used during training to train a Fast R-CNN network.

DisentagleNetworkLayer implements a Caffe Python layer.
"""

import caffe
import numpy as np
import yaml

class DisentagleNetworkLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        if self._ssd:
            self.forward_ssd(bottom,top)
        else:
            self.forward_faster_rcnn(bottom,top)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

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
