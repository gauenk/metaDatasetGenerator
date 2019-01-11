import numpy as np
import pickle,sys,os
from datasets import ds_utils
from core.config import cfg,cfgData,load_tp_fn_record_path
import os.path as osp

cfg._DEBUG.datasets.evaluators.cls_utils = False

def cocoID_to_base(cocoID):
    if "COCO" in cocoID:
        index = int(re.match(".*(?P<id>[0-9]{12})",cocoID).groupdict()['id'])
    else:
        index = cocoID
    return index

def parse_anno(filename,load_annotation,classes):
    # convert if coco
    index = cocoID_to_base(filename)
    anno = load_annotation(index)
    objects = []
    for idx,gt_classes in enumerate(anno['gt_classes']):
        obj_struct = {}
        obj_struct['anno'] = filename
        obj_struct['cls'] = int(anno['gt_classes'][idx])
        obj_struct['cls_id'] = idx
        objects.append(obj_struct)
    return objects

def load_cls_groundTruth(classname,cachedir,imagesetfile,annopath,annoReader,indexToCls):
    # first load gt
    if not osp.isdir(cachedir):
        os.mkdir(cachedir)
    print(cachedir)
    cachefile = osp.join(cachedir, 'annots.pkl')

    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    
    """
    ^I think this is the same as the "image_index" in the imdb
    """
    
    """
    _ I think we can get the annotation from 
    """

    if not osp.isfile(cachefile):
        # load annots
        annos = {}
        for i, imagename in enumerate(imagenames):
            annos[imagename] = parse_anno(imagename,annoReader,indexToCls)
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            pickle.dump(annos, f)
    else:
        # load
        print("cachefile @ {:s}".format(cachefile))
        with open(cachefile, 'r') as f:
            annos = pickle.load(f)

    return imagenames, annos

def load_groundTruth(classname,cachedir,imagesetfile,annopath,_load_annotation,_classes):
    if cfg.SUBTASK == 'tp_fn':
        records = ds_utils.loadEvaluationRecords(classname,cfg.TP_FN_RECORDS_PATH,cfg.DATASETS)
        imagenames,probs = zip(*records.items())
        probsOrAnnos = probs
    elif cfg.SUBTASK in ["default","al_subset"]:
        imagenames,annos = load_cls_groundTruth(classname,cachedir,imagesetfile,annopath,_load_annotation,_classes)
        probsOrAnnos = annos
    return imagenames, probsOrAnnos

def extractClassGroundTruth(imagenames,annos,classname):
    if cfg.SUBTASK == 'tp_fn' and cfg.DATASETS.HAS_BBOXES:
        image_ids,class_probs,npos,nneg = extractImageIdsAndProbs(imagenames,annos,True)
        class_probs_dict = dict(zip(image_ids,class_probs))
    elif cfg.SUBTASK in ["default","al_subset"] or not cfg.DATASETS.HAS_BBOXES:
        image_ids,class_probs,npos,nneg = extractImageIdsAndProbs(imagenames,annos,False)
        class_probs_dict = dict(zip(image_ids,class_probs))
    else:
        print("ERROR: unknown cfg.Subtask {} in [lib/datasets/evaluate/cls_utils.py: extractClassGroundTruth]".format(cfg.SUBTASK))
        sys.exit()
    return image_ids,class_probs_dict,npos,nneg

def loadModelCls(detpath,classname,dataset_augmentations):
    # read the file with the model detections
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    # interpret each line from the file
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    if cfg.SUBTASK == "tp_fn": items = np.array([ [ round(float(y),3) for y in x[1:] ] for x in splitlines])
    elif cfg.SUBTASK in ["default","al_subset"]:
        item_index = 1
        if len(dataset_augmentations) != 0: item_index = 2
        items = np.array([ int(x[item_index]) for x in splitlines]) # index @ 1 is the guessed class
    else: print("ERROR: unknown cfg.Subtask {} in [lib/datasets/evaluate/cls_utils.py: loadModelCls]".format(cfg.SUBTASK))
    # TODO: asses impact of model_probs/model_output_cls for evaluation_v1/v2

    # sort the detections by confidence for scoring    
    if cfg.SUBTASK == "tp_fn":
        if items.shape[0] == 1 or len(items.shape) == 1:
            sorted_ind = np.argsort(-items)
            sorted_scores = np.sort(-items)
            image_ids = [image_ids[x] for x in sorted_ind]

    return image_ids, items

def extractImageIdsAndProbs(imagenames,annos,flatten):
    #print("extractImageIdsAndProbs")
    if type(annos) is dict:
        return extractImageIdsAndProbs_annoDict(imagenames,annos,flatten)
    elif type(annos) is list or type(annos) is tuple:
        return extractImageIdsAndProbs_annoList(imagenames,annos,flatten)
    
def extractImageIdsAndProbs_annoDict(imagenames,annos,flatten):
    #print("extractImageIdsAndProbs_annoDict")
    npos = 0
    nneg = 0
    gt_classes = []
    for image_id in imagenames:
        anno = annos[image_id]
        if flatten is False and cfg.SUBTASK in ["default","al_subset"]:
            gt_classes.append(anno[0]['cls'])
    return imagenames,np.array(gt_classes),npos,nneg
    
def extractImageIdsAndProbs_annoList(imagenames,annos,flatten):
    #print("extractImageIdsAndProbs_annoList")
    npos = 0
    nneg = 0
    gt_classes = []
    image_ids = []
    for image_id,anno in zip(imagenames,annos):
        n_image_items = 0
        for item in anno:
            if flatten:
                image_id_new = image_id + "_{}".format(str(n_image_items))
                n_image_items += 1
            else:
                image_id_new = str(image_id)
            image_ids.append(image_id_new)
            gt_classes.append(item)
            npos += (item == 1)
            nneg += (item == 0)
    return image_ids,np.array(gt_classes),npos,nneg


def compute_metrics(ovthresh,image_ids,model_probs,gt_classes,num_classes,class_convert):
    if cfg.SUBTASK == "tp_fn":
        return compute_metrics_v1(ovthresh,image_ids,model_probs,gt_classes,class_convert)
    elif cfg.SUBTASK in ["default","al_subset"]:
        return compute_metrics_v2(ovthresh,image_ids,model_probs,gt_classes,num_classes,class_convert)
    else:
        print("ERROR: unknown cfg.Subtask {} in [lib/datasets/evaluate/cls_utils.py: loadModelCls]".format(cfg.SUBTASK))
        sys.exit()

def compute_metrics_v2(ovthresh,image_ids,model_output_cls,gt_classes,num_classes,class_convert):
    print("[cls_utils.py: compute_metrics_v2]")
    nd = len(image_ids)
    print("nd: {}".format(nd))
    correct = np.zeros((nd,len(ovthresh))) # "false negative"
    records = {}
    for d in range(nd):
        CLS_GT = gt_classes[image_ids[d]] # should be size 1
        for idx in range(len(ovthresh)):
            # thresh = ovthresh[idx]
            # # get a guess
            # guess = 0
            # if prob >= thresh: guess = 1
            # record if correct
            guess = class_convert[model_output_cls[d]]
            # if idx == 0:
            #     print(guess,CLS_GT)
            if guess == CLS_GT: correct[d,idx] = 1
        records[image_ids[d]] = [1*np.any(correct[d,:]==1)]
    print(np.sum([sample for sample in records.values()]))
    saveRecord(records)
    acc = np.mean(correct,axis=0)
    print(acc)
    print("Accuracy: {:.6f}".format(acc[0]))
    print("DONE")
    sys.exit()
    return correct,None,None,None

def saveRecord(objToSave,suffix='cls'):
    saveDir = osp.join(cfg.TP_FN_RECORDS_PATH,cfg.DATASETS.CALLING_DATASET_NAME)
    if cfg.TP_FN_RECORDS_WITH_IMAGESET:
        datasetDest = "{}-{}-default".format(cfg.DATASETS.CALLING_DATASET_NAME,cfg.DATASETS.CALLING_IMAGESET_NAME)
        if cfg.IMAGE_ROTATE != 0 and cfg.IMAGE_ROTATE != False:
            datasetDest = "{}-{}-default-rot{}".format(cfg.DATASETS.CALLING_DATASET_NAME,cfg.DATASETS.CALLING_IMAGESET_NAME,cfg.IMAGE_ROTATE)
        saveDir = osp.join(cfg.TP_FN_RECORDS_PATH,datasetDest)
    if not osp.isdir(saveDir):
        os.makedirs(saveDir)
    print(saveDir)
    print(cfg.TP_FN_RECORDS_WITH_KEYS)
    if cfg.TP_FN_RECORDS_WITH_KEYS:
        savePath = osp.join(saveDir,"records_{}_{}.pkl".format(suffix,cfg.TEST_NET.NET_PATH))
        print(savePath)
        with open(savePath,'wb') as f:
            pickle.dump(objToSave,f)
    else:
        print("IN THE ELSE")
        print(cfg.TEST_NET.NET_PATH)
        new_record = remove_keys_from_activity_vector_and_ravel(objToSave)
        savePath = osp.join(saveDir,"records_{}_{}.npy".format(suffix,cfg.TEST_NET.NET_PATH))
        print(savePath)
        np.save(savePath,new_record)

def remove_keys_from_activity_vector_and_ravel(record_dict):
    new_record = [None for _ in range(len(record_dict))]
    sorted_keys = sorted(record_dict.keys()) # ensures consistency
    for index,key in enumerate(sorted_keys):
        new_record[index] = record_dict[key][0]
    new_record = np.array(new_record)
    return new_record

def compute_metrics_v1(ovthresh,image_ids,model_probs,gt_classes,class_convert):
    print("[cls_utils.py: compute_metrics_v1]")
    nd = len(image_ids)
    tp = np.zeros((nd,len(ovthresh))) # "true positive"
    tn = np.zeros((nd,len(ovthresh))) # "true negative"
    fp = np.zeros((nd,len(ovthresh))) # "false positive"
    fn = np.zeros((nd,len(ovthresh))) # "false negative"
    # print(len(gt_classes.keys()))
    # print(gt_classes)
    # print(image_ids)

    for d in range(nd):
        CLS_GT = gt_classes[image_ids[d]] # should be size 1
        if model_probs[d].size == 1:
            prob = model_probs[d]
        else:
            prob = model_probs[d, :].astype(float) # should be size 1
        # if cfg._DEBUG.datasets.evaluators.cls_utils: print("[compute_TP_FP]",bb,BBGT)
        
        for idx in range(len(ovthresh)):
            thresh = ovthresh[idx]
            assert (0 < thresh and thresh < 1), "thresh in (0,1)"

            guess = 0
            if prob >= thresh: guess = 1

            if guess == CLS_GT and guess == 1: tp[d,idx] = 1
            if guess == CLS_GT and guess == 0: tn[d,idx] = 1
            if guess != CLS_GT and guess == 1: fp[d,idx] = 1
            if guess != CLS_GT and guess == 0: fn[d,idx] = 1

    tpr = tp.sum(axis=0) / (tp.sum(axis=0) + fn.sum(axis=0))
    tnr = tn.sum(axis=0) / (tn.sum(axis=0) + fp.sum(axis=0))
    ppv = tp.sum(axis=0) / (tp.sum(axis=0) + fp.sum(axis=0))
    npv = tn.sum(axis=0) / (tn.sum(axis=0) + fn.sum(axis=0))
    acc = ( tp.sum(axis=0) + tn.sum(axis=0) ) /\
          ( tp.sum(axis=0) + tn.sum(axis=0) + fp.sum(axis=0) + fn.sum(axis=0) )
    print("TPR: {:.6f}".format(tpr[0]))
    print("TNR: {:.6f}".format(tnr[0]))
    print("PPV: {:.6f}".format(ppv[0]))
    print("NPV: {:.6f}".format(npv[0]))
    print("ACC: {:.6f}".format(acc[0]))
    sys.exit()
    return tp, tn, fp, fn










