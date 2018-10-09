import numpy as np
from datasets import ds_utils
from core.config import cfg,cfgData

def load_groundTruth(classname):
    if cfg.TEST.CLASSIFICATION.TASK == 'tp_fn':
        records = ds_utils.loadEvaluationRecords(classname)
        imagenames,probs = zip(*records.items())
    return imagenames, probs

def extractClassGroundTruth(imagenames,probs,classname):
    if cfg.TEST.CLASSIFICATION.TASK == 'tp_fn':
        image_ids,class_probs,npos,nneg = flattenImageIdsAndProbs(imagenames,probs)
        class_probs_dict = dict(zip(image_ids,class_probs))
    return image_ids,class_probs_dict,npos,nneg


def loadModelCls(detpath,classname):
    # read the file with the model detections
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    # interpret each line from the file
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    probs = np.array([ [ round(float(y),3) for y in x[1:] ] for x in splitlines])

    # sort the detections by confidence for scoring    
    if probs.shape[0] == 1 or len(probs.shape) == 1:
        sorted_ind = np.argsort(-probs)
        sorted_scores = np.sort(-probs)
        image_ids = [image_ids[x] for x in sorted_ind]
    return image_ids, probs

def flattenImageIdsAndProbs(imagenames,probs):
    npos = 0
    nneg = 0
    class_probs = []
    image_ids = []
    for image_id,tp_fn_list in zip(imagenames,probs):
        n_image_items = 0
        for item in tp_fn_list:
            image_id_new = image_id + "_{}".format(str(n_image_items))
            image_ids.append(image_id_new)
            class_probs.append(item)
            npos += (item == 1)
            nneg += (item == 0)
            n_image_items += 1
    return image_ids,np.array(class_probs),npos,nneg

def compute_metrics(ovthresh,image_ids,model_probs,gt_class_probs):
    nd = len(image_ids)
    tp = np.zeros((nd,len(ovthresh))) # "true positive"
    tn = np.zeros((nd,len(ovthresh))) # "true negative"
    fp = np.zeros((nd,len(ovthresh))) # "false positive"
    fn = np.zeros((nd,len(ovthresh))) # "false negative"

    for d in range(nd):
        CLS_GT = gt_class_probs[image_ids[d]] # should be size 1
        if model_probs[d].size == 1:
            prob = model_probs[d]
        else:
            prob = model_probs[d, :].astype(float) # should be size 1
        if cfg._DEBUG.datasets.evaluators.bbox_utils: print("[compute_TP_FP]",bb,BBGT)
        
        for idx in range(len(ovthresh)):
            thresh = ovthresh[idx]
            assert (0 < thresh and thresh < 1), "thresh in (0,1)"

            guess = 0
            if prob >= thresh: guess = 1

            if guess == CLS_GT and guess == 1: tp[d,idx] = 1
            if guess == CLS_GT and guess == 0: tn[d,idx] = 1
            if guess != CLS_GT and guess == 1: fp[d,idx] = 1
            if guess != CLS_GT and guess == 0: fn[d,idx] = 1


    return tp, tn, fp, fn










