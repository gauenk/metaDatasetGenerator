"""Test a network with a DataLoader."""

from core.config import cfg, get_output_dir
from datasets.data_loader import DataLoader
from core.test_utils.active_learning_report import activeLearningReportAppendActivationValueData
from core.test_utils.agg_model_output import aggregateModelOutput
from core.test_utils.agg_activations import aggregateActivations
from easydict import EasyDict as edict

from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
from utils.misc import save_image_with_border
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob,blob_list_im,save_blob_list_to_file
from datasets.ds_utils import cropImageToAnnoRegion
from cls_data_layer.minibatch import get_minibatch as cls_get_minibatch
# from alcls_data_layer.minibatch import get_minibatch as alcls_get_minibatch
# from aim_data_layer.minibatch import get_minibatch as aim_get_minibatch
import os
import matplotlib.pyplot as plt

cfg._DEBUG.core.test = False

def pre_process_model_inputs(net,modelInputs,image_scales,image_id,layer_names_for_activations):

    if cfg.TEST.INPUTS.IM_INFO:
        im_info = np.array([[im_blob.shape[2], im_blob.shape[3], image_scales[0]]],dtype=np.float32)
        modelInputs['im_info'] = im_info

    if cfg.TEST.INPUTS.RESHAPE: #(if cfg.SSD is False:)
        for input_name,input_value in modelInputs.items():
            net.blobs[input_name].reshape(*(input_value.shape))

    if cfg._DEBUG.core.test:
        im_shape = forward_kwargs["data"].shape
        rimg = blob_list_im(forward_kwargs["data"])
        fn = "input_image_{}_{}.png".format(cfg.IMAGE_ROTATE,image_id)
        #save_image_with_border(fn,rimg[0],rotation=im_rotates[0])
    if len(layer_names_for_activations) > 0:
        modelInputs['blobs'] = layer_names_for_activations

def post_process_model_outputs(net,modelOutput,image_scales,layer_names_for_activations):

    if cfg.TEST.OBJ_DET.HAS_RPN and cfg.SSD is False and cfg.TASK == 'object_detection':
        assert len(image_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / image_scales[0]
    elif cfg.SSD is True and cfg.TASK == 'object_detection':
        #assert len(image_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['detection_out'].data.copy()
        # unscale back to raw image space
        boxes = rois[0,0,:, 3:] * cfg.SSD_img_size
        boxes[:,0] = boxes[:,0] / image_scales[0][0]
        boxes[:,1] = boxes[:,1] / image_scales[0][1]
        boxes[:,2] = boxes[:,2] / image_scales[0][0]
        boxes[:,3] = boxes[:,3] / image_scales[0][1]

    if cfg.TEST.OBJ_DET.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    elif cfg.SSD is False or cfg.TASK == 'classification':
        # use softmax estimated probabilities
        scores = modelOutput[cfg.CLS_PROBS] # default is "cls_prob"
    elif cfg.SSD is True and cfg.TASK == 'object_detection':
        scores = np.zeros((len(boxes),201))
        for row in range(len(rois[0,0,:,2])):
            scores[row,rois[0,0,row, 1].astype(int)] = rois[0,0,row, 2]

    pred_boxes = []
    if cfg.TEST.OBJ_DET.BBOX_REG and cfg.SSD is False and cfg.TASK == 'object_detection':
        # Apply bounding-box regression deltas
        box_deltas = modelOutput['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    elif cfg.SSD is True and cfg.TASK == 'object_detection':
        box_deltas = np.zeros((len(boxes),804)) ##CHANGE IF DIFF NUM OF CLASS
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    elif cfg.TASK == 'object_detection':
        # Simply repeat the boxes, once for each class
        print("\n\n\n\nDANGER DANGER! WE SHOULD NOT USE THIS\n\n\n\n\n")
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.OBJ_DET.DEDUP_BOXES > 0 and not cfg.TEST.OBJ_DET.HAS_RPN and cfg.TASK == 'object_detection':
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    activitions = {}
    if len(layer_names_for_activations) > 0:
        for blob_name in layer_names_for_activations:
            # if blob_name not in activity_vectors.keys(): activity_vectors[blob_name] = []
            # above is always true since dict should start empty
            # here we know it is only one image id.
            activitions[blob_name] = modelOutput[blob_name].astype(np.float32, copy=True)
    return scores, pred_boxes, activitions



def im_detect(net, modelInputs, image_scales, layer_names_for_activations, image_id=""):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        imgBlob (ndarray): color image blob to test (in BGR order)
        scale (ndarray): 1x1 or 1x2 if one or two sides were fixed in image resize
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    pre_process_model_inputs(net,modelInputs,image_scales,image_id,layer_names_for_activations)
    modelOutputs = net.forward(**modelInputs)
    scores, pred_boxes, activitions = post_process_model_outputs(net,modelOutputs,image_scales,layer_names_for_activations)

    return scores, pred_boxes, activitions

def check_config_for_error():
    # handle region proposal network
    if cfg.TEST.OBJ_DET.HAS_RPN is False and cfg.TASK == 'object_detection':
        raise ValueError("We can't handle rpn correctly. See [box_proposals] in original faster-rcnn code.")
    

def test_net(net, imdb, max_dets_per_image=100, thresh=1/80., vis=False, al_net=None):
    """Test a model on an repo_imdb"""

    check_config_for_error()

    # is this weirdness gone now?
    roidb = imdb.roidb #imdb weirdness; we need to load this before we get set values 
    output_dir = get_output_dir(imdb, net)
    correctness_records = [] # load records in future for aim/alcls

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    
    # config for loading a sample
    ds_loader = imdb.create_data_loader(cfg,correctness_records,al_net)

    # set mange ds_loader for net 
    alReport = activeLearningReportAppendActivationValueData(net,imdb,cfg.ACTIVE_LEARNING,False)    # TODO: add RECORDS BOOL
    aggModelOutput = aggregateModelOutput(imdb,ds_loader.num_samples,output_dir,cfg.TASK,thresh,cfg.TEST.OBJ_DET.NMS,max_dets_per_image,vis,cfg)
    aggActivations = aggregateActivations(cfg.ACTIVATION_VALUES,cfg.GET_SAVE_ACTIVITY_VECTOR_BLOBS_DIR())

    # main loop
    for imageBlob,scales,sample,index in ds_loader.dataset_generator(imdb.data_loader_config,load_as_blob=True):

        _t['im_detect'].tic()
        scores, boxes, activitions = im_detect(net, imageBlob, scales, layer_names_for_activations, sample['image_id'])
        _t['im_detect'].toc()

        _t['misc'].tic()
        model_output = {"scores":scores,"boxes":boxes,"activitions":activitions}

        aggActivations.aggregate(model_output,sample['image_id'])
        aggModelOutput.aggregate(model_output,index)
        alReport.record(model_output,sample['image_id'])

        _t['misc'].toc()
        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s').format(index + 1, ds_loader.num_samples, _t['im_detect'].average_time,_t['misc'].average_time)

    aggActivationValues.save(net.name) # TODO: might be something else...
    aggModelOutput.save(ds_loader.dataset_augmentation.configs)

    print('Evaluating detections')
    imdb.evaluate_detections(aggModelOutput.results, output_dir, ds_loader)
    

