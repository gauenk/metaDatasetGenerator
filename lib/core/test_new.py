"""Test a network with a DataLoader."""

from core.config import cfg, get_output_dir
from datasets.data_loader import DataLoader
from core.test_utils.active_learning_report import activeLearningReportAppendActivationData
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

def post_process_model_outputs_for_detection(net,modelOutput,image_scales,layer_names_for_activations):

    if cfg.TEST.OBJ_DET.HAS_RPN and cfg.SSD is False:
        assert len(image_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / image_scales[0]
    elif cfg.SSD is True:
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
    elif cfg.SSD is False:
        # use softmax estimated probabilities
        scores = modelOutput[cfg.CLS_PROBS] # default is "cls_prob"
    elif cfg.SSD is True:
        scores = np.zeros((len(boxes),201))
        for row in range(len(rois[0,0,:,2])):
            scores[row,rois[0,0,row, 1].astype(int)] = rois[0,0,row, 2]

    pred_boxes = []
    if cfg.TEST.OBJ_DET.BBOX_REG and cfg.SSD is False:
        # Apply bounding-box regression deltas
        box_deltas = modelOutput['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    elif cfg.SSD is True:
        box_deltas = np.zeros((len(boxes),804)) ##CHANGE IF DIFF NUM OF CLASS
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        print("\n\n\n\nDANGER DANGER! WE SHOULD NOT USE THIS\n\n\n\n\n")
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.OBJ_DET.DEDUP_BOXES > 0 and not cfg.TEST.OBJ_DET.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores,pred_boxes

def post_process_model_outputs(net,modelOutput,image_scales,layer_names_for_activations):

    if cfg.TASK == 'object_detection':
        scores,pred_boxes = post_process_model_outputs_for_detection(net,modelOutput,image_scales,layer_names_for_activations)        
    elif cfg.TASK == 'classification' or cfg.TASK == 'regression':
        pred_boxes = []
        scores = modelOutput[cfg.CLS_PROBS] # default is "cls_prob"

    activations = {}
    if len(layer_names_for_activations) > 0:
        for blob_name in layer_names_for_activations:
            # if blob_name not in activity_vectors.keys(): activity_vectors[blob_name] = []
            # above is always true since dict should start empty
            # here we know it is only one image id.
            activations[blob_name] = modelOutput[blob_name].astype(np.float32, copy=True)
    return scores, pred_boxes, activations



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
    scores, pred_boxes, activations = post_process_model_outputs(net,modelOutputs,image_scales,layer_names_for_activations)

    return scores, pred_boxes, activations

def model_inference(net,ds_loader,aggModelOutput,aggActivations,alReport):
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    layer_names_for_activations = cfg.ACTIVATIONS.LAYER_NAMES
    # main loop
    for imageBlob,scales,sample,index in ds_loader.dataset_generator(load_as_blob=True):

        _t['im_detect'].tic()
        scores, boxes, activations = im_detect(net, imageBlob, scales, layer_names_for_activations, sample['image_id'])
        _t['im_detect'].toc()

        _t['misc'].tic()
        model_output = {"scores":scores,"boxes":boxes,"activations":activations}

        aggModelOutput.aggregate(model_output,index)
        aggActivations.aggregate(activations,sample['image_id'])
        alReport.record(model_output,sample['image_id'])

        _t['misc'].toc()
        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s').format(index + 1, ds_loader.num_samples, _t['im_detect'].average_time,_t['misc'].average_time)

    aggModelOutput.save()
    aggActivations.save()

def test_net(net, imdb, max_dets_per_image=100, thresh=1/80., vis=False, al_net=None, new_cache=False):
    """Test a model on an repo_imdb"""

    # is this weirdness gone now?
    roidb = imdb.roidb #imdb weirdness; we need to load this before we get set values 
    output_dir = get_output_dir(imdb, net)
    correctness_records = [] # load records in future for aim/alcls
    
    # config for loading a sample
    ds_loader = imdb.create_data_loader(cfg,correctness_records,al_net)
    if cfg.modelInfo.siamese:
        dslcfg = ds_loader.dataset_loader_config
        dslcfg.siamese = True
        data_index = dslcfg.load_fields.index("data")
        dslcfg.load_fields[data_index] += "_0"
        dslcfg.load_fields.append("data_1")

    # set mange ds_loader for net 
    aggActivations = aggregateActivations(cfg.ACTIVATIONS,cfg,imdb)
    alReport = activeLearningReportAppendActivationData(net,imdb,cfg.ACTIVE_LEARNING,False) # TODO: add RECORDS BOOL
    aggModelOutput = aggregateModelOutput(imdb,ds_loader.num_samples,output_dir,cfg.TASK,thresh,cfg.TEST.OBJ_DET.NMS,max_dets_per_image,vis,cfg)


    """
    1.) have we evaluated the model's outputs? If not, keep going.
        -> we might not need to load the model's output (possibly large in memory) 
    """
    evaluataion = None # imdb.evaluation_results(ds_loader, aggModelOutput, output_dir)
    if evaluataion:
        imdb.evaluate_model_inference(ds_loader, aggModelOutput, output_dir)

    """
    2.) have we already computed the model's outputs? If not, compute them.
        -> we might not need to recompute the model's output (expensive in time and memory)
    """
    print("PRETEST_NEW")
    check_activations_loaded = aggActivations.load_and_verify(cfg.ACTIVATIONS.LAYER_NAMES)
    print("POSTTEST_NEW")
    print("PRETEST_NEW@")
    eval_results = aggModelOutput.load()
    print("POSTTEST_NEW#")
    if eval_results is None or check_activations_loaded is False or new_cache is True:
        aggModelOutput.clear()
        aggActivations.save_bool = cfg.ACTIVATIONS.SAVE_BOOL
        model_inference(net,ds_loader,aggModelOutput,aggActivations,alReport)

    """
    3.) we are forced to evaluate the results now.
    """
    #print(aggModelOutput.results)
    print('Evaluating detections')
    imdb.evaluate_model_inference(ds_loader, aggModelOutput, output_dir,activations=aggActivations)
    

