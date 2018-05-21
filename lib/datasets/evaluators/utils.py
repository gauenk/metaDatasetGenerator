
import os.path as osp
import cPickle

import xml.etree.ElementTree as ET
import numpy as np

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename+".xml")
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['anno'] = filename
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def load_groundTruth(cachedir,imagesetfile,annopath,annoReader):
    # first load gt
    if not osp.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = osp.join(cachedir, 'annots.pkl')

    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not osp.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        print("cachefile @ {:s}".format(cachefile))
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    return imagenames, recs


def extractClassGroundTruth(imagenames,recs,classname):
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
    return class_recs,npos

def loadModelDets(detpath,classname):
    # read the file with the model detections
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    # interpret all values from the file
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([round(float(x[1]),3) for x in splitlines])
    BB = np.array([[round(float(z),1) for z in x[2:]] for x in splitlines])

    # sort the detections by confidence for scoring    
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    if len(BB) == 0:
        print("no predicted boxes")
        raise ValueError("There are no detections from the model for class {}".format(classname))
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    return image_ids, BB

def compute_TP_FP(ovthresh,image_ids,BB,class_recs):
    nd = len(image_ids)
    tp = np.zeros((nd,len(ovthresh)))
    fp = np.zeros((nd,len(ovthresh)))
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        # if(sorted_scores[d] >= -0.5):
        #     continue
        #print(sorted_scores[d],sorted_scores[d] < -0.0)
        inside_any = False
        for idx in range(len(ovthresh)):
            if ovmax > ovthresh[idx]:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        inside_any = True
                        tp[d,idx] = 1.
                        #print("tp")
                    else:
                        fp[d,idx] = 1.
                        #print("fp")
            else:
                fp[d,idx] = 1.
                #print("fp")

        if inside_any is True:
            R['det'][jmax] = 1


    return tp, fp

def compute_REC_PREC_AP(tp,fp,npos,ovthresh,classname,use07=False,viz=False):
    rec = np.zeros((len(fp),len(ovthresh)))
    prec = np.zeros((len(fp),len(ovthresh)))
    ap = np.zeros(len(ovthresh))
    for idx in range(len(ovthresh)):
        # compute precision recall
        _fp = np.cumsum(fp[:,idx])
        _tp = np.cumsum(tp[:,idx])
        rec[:,idx] = _tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec[:,idx] = _tp / np.maximum(_tp + _fp, np.finfo(np.float64).eps)
        #ap = bbox_ap(rec, prec, use_07_metric)
        ap[idx] = bbox_ap(rec[:,idx], prec[:,idx], classname, use_07_metric = use07, viz=viz)
    return rec, prec, ap

def bbox_ap(rec, prec, clsnm,use_07_metric=False,viz=False):
    """ ap = bbox_ap(rec, prec, [use_07_metric])
    Compute BBOX AP given precision and recall.
    If use_07_metric is true, uses the
    BBOX 07 11 point method (default:False).
    """
    if viz:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

    use_07_metric = True

    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        plt.plot(mrec,mpre,"g.")

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])


        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        if viz:
            plt.plot(mrec,mpre,"ro",mrec,ap,"g^")
            plt.plot(mrec,mpre,"r+")
            plt.tight_layout()
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title("AP{}".format(ap))
            plt.savefig(clsnm + "_apPlt.png")        

    return ap


