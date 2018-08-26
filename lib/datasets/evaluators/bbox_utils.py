import pickle,utils,os,re,sys,Image,ImageDraw,cv2
import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
from core.config import cfg
from utils.misc import getRotationScale,toRadians,save_image_with_border,getRotationInfo,npBoolToUint8,save_image_of_overlap_bboxes

#cfg._DEBUG.datasets.evaluators.bbox_utils = True

def cocoID_to_base(cocoID):
    if "COCO" in cocoID:
        index = int(re.match(".*(?P<id>[0-9]{12})",cocoID).groupdict()['id'])
    else:
        index = cocoID
    return index

def parse_rec(filename,load_annotation,classes):

    # convert if coco
    index = cocoID_to_base(filename)
    recs = load_annotation(index)
    objects = []
    for idx,bbox in enumerate(recs['boxes']):
        obj_struct = {}
        obj_struct['anno'] = filename
        obj_struct['name'] = recs['gt_classes'][idx]
        try:
            if int(obj_struct['name']) is not None:
                obj_struct['name'] = classes[obj_struct['name']]
        except:
            pass
        obj_struct['difficult'] = 0
        obj_struct['bbox'] = bbox
        obj_struct['bbox_id'] = idx
        objects.append(obj_struct)
    return objects

def load_groundTruth(cachedir,imagesetfile,annopath,annoReader,indexToCls):
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
            recs[imagename] = parse_rec(imagename,annoReader,indexToCls)
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            pickle.dump(recs, f)
    else:
        # load
        print("cachefile @ {:s}".format(cachefile))
        with open(cachefile, 'r') as f:
            recs = pickle.load(f)

    return imagenames, recs


def extractClassGroundTruth(imagenames,recs,classname):

    # printing infor for debug
    if False:
        for fn,objs in recs.items():
            print("="*100)
            print(fn)
            for obj in objs:
                print(obj['name'])

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        bbox_id = None
        if len(R) > 0:
            if 'bbox_id' in R[0].keys(): bbox_id = np.array([x['bbox_id'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det,
                                 'bbox_id':bbox_id,
                                 'count':0}
    return class_recs,npos

def transformImageId(imageID):
    # convert if coco
    index = cocoID_to_base(imageID)
    return str(index)

def loadModelDets(detpath,classname):
    # read the file with the model detections
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    # interpret each line from the file
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
        if cfg._DEBUG.datasets.evaluators.bbox_utils: print("[compute_TP_FP]",bb,BBGT)
        if BBGT.size == 0: print("zero @ {}".format(d))
        
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
            if cfg._DEBUG.datasets.evaluators.bbox_utils: print("ovmax",ovmax)


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

def compute_TP_FP_rotation(ovthresh,image_ids,BB,class_recs,rotations):
    nd = len(image_ids)
    tp = np.zeros((nd,len(ovthresh)))
    fp = np.zeros((nd,len(ovthresh)))
    for d in range(nd):
        rotation = rotations[image_ids[d]][0] # the zero index comes from using only one image in a given batch
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        if cfg._DEBUG.datasets.evaluators.bbox_utils: print("rotation",rotation)
        if BBGT.size == 0: print("zero @ {}".format(d))
        
        if BBGT.size > 0:
            # compute overlaps

            if len(rotation) > 0:
                pimgs = polygonImageList(BBGT,rotation)
                if cfg._DEBUG.datasets.evaluators.bbox_utils: print("after polygonimagelist",pimgs.shape)
                overlaps = computeIOU_from_PolygonList(pimgs,bb,rotation,d,image_ids[d])
            else:
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
            if cfg._DEBUG.datasets.evaluators.bbox_utils: print("ovmax",ovmax)
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
    show = [True,False,False]
    for idx in range(len(ovthresh)):
        # compute precision recall
        _fp = np.cumsum(fp[:,idx])
        _tp = np.cumsum(tp[:,idx])
        rec[:,idx] = _tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec[:,idx] = _tp / np.maximum(_tp + _fp, np.finfo(np.float64).eps)
        #ap = bbox_ap(rec, prec, use_07_metric)
        ap[idx] = bbox_ap(rec[:,idx], prec[:,idx], classname, use_07_metric = use07, viz=show[idx])
    return rec, prec, ap

def record_TP_FP_IMAGE_AND_BBOX_ID(tp,fp,class_recs,image_ids,classname):
    # TODO: write this function. :)
    print("-=-=-=- class: {} -=-=-=-".format(classname))
    print("WARNING: only prints TP and FN detections. That is, not False Positives.")
    image_ids = set(image_ids)
    nd = len(image_ids) # we only need to see each one once
    for d,image_id in enumerate(image_ids):
        R = class_recs[image_id]
        if cfg._DEBUG.datasets.evaluators.bbox_utils: print(R['det'],image_id)
        # if R['bbox'].size == 0: continue
        # isCorrect = tp[d,0]
        # if isCorrect:
        #     print(d,tp[d,0],image_ids[d],R['bbox_id'][R['count']])
        # else:
        #     print(d,tp[d,0],image_ids[d])
        # R['count'] += tp[d,0]
    return True
    

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

    use_07_metric = False

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
        if viz:
            plt.clf()
            plt.cla()
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
            if cfg._DEBUG.datasets.evaluators.bbox_utils:
                print(mrec.shape)
                print(mpre.shape)
                print(ap)
                print(ap.shape)
            #plt.plot(mrec,mpre,"ro",mrec,ap,"g^")
            plt.plot(mrec,mpre,"r+")
            plt.tight_layout()
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title("AP {}".format(ap))
            fn = clsnm
            if cfg.ROTATE_IMAGE:
                fn += "_" + str(cfg.ROTATE_IMAGE)
            fn += "_apPlt.png"
            plt.savefig(fn)
        return ap


def polygonImageList(box_list,rotation):
    #img_size = polygonImage_getImageSize(box_list)
    img_size = (rotation[2],rotation[1],3)
    pimgs = np.zeros((len(box_list),) + img_size)
    for idx,box in enumerate(box_list):
        pimg = polygonImage(box.reshape(2,2),img_size,rotation,idx)
        pimgs[idx,:,:,:] = pimg
    return pimgs

def polygonImage_getImageSize(box_list):
    min_size = np.max(box_list)
    return (int(min_size),int(min_size),3)

def polygonImage(box,img_size,rotation,num):
    """
    -> the image shape is contrained such that the original or translated bounding box will not be outside of the image. That is, the image size should be *at least* greater than the bottom right of the bounding box.
    -> for now, use x2 the bottom right of the bounding box as the image size.
    """
    points = ( tuple(box[0,:]),
                 (box[1,0], box[0,1]),
                 tuple(box[1,:]),
                 (box[0,0], box[1,1])
               )
    np_zeros = np.zeros(img_size).astype(np.uint8)
    zero_img = Image.fromarray(np_zeros)
    draw = ImageDraw.Draw(zero_img)
    draw.polygon(points,fill="white")
    box_img = np.array(zero_img,dtype=np.uint8)
    angle = rotation[0]
    cols = rotation[1]
    rows = rotation[2]
    M,scale = getRotationInfo(angle,cols,rows)
    post_warp = cv2.warpAffine(box_img,M,(cols,rows),scale).astype(np.uint8)

    if cfg._DEBUG.datasets.evaluators.bbox_utils:
        zero_img.save("./only_polygon_{}.png".format(num))
        print("angle",angle)
        print("M",M)
        print("scale",scale)
        cv2.imwrite("./only_poly_rotated_{}.png".format(num),post_warp)

    return post_warp.astype(np.uint8)

def computeIOU_from_PolygonList(pimgs,bb,rotation,guessNumber,imgId):
    overlaps = [ None for _ in range(pimgs.shape[0]) ]
    img_size = pimgs.shape[1:]
    rotation_none = [ None for _ in range(len(rotation)) ]
    rotation_none[1:] = rotation[1:]
    rotation_none[0] = 0
    guessBoxImg = polygonImage(bb.reshape(2,2),img_size,rotation_none,-1)
    if cfg._DEBUG.datasets.evaluators.bbox_utils:
        print("[computeIOU_from_PolygonList]",rotation)
        save_image_with_border("./guess.png",guessBoxImg)
    for idx,pimg in enumerate(pimgs):
        inter = np.bitwise_and(guessBoxImg.astype(np.bool),pimg.astype(np.bool))
        union = np.bitwise_or(guessBoxImg.astype(np.bool),pimg.astype(np.bool))
        overlap = np.sum(inter) / float(np.sum(union))
        overlaps[idx] = overlap
        if cfg._DEBUG.datasets.evaluators.bbox_utils:
            save_image_with_border("./guess_AND_{}.png".format(idx),npBoolToUint8(inter),
                                   rotation=rotation)
            #save_image_with_border("./gt_{}.png".format(idx),pimg,rotation=rotation)
            cv2.imwrite("./gt_{}.png".format(idx),pimg)
            save_image_of_overlap_bboxes("./show_overlap_img_{}_guess_{}_gt_{}.png".format(imgId,guessNumber,idx),guessBoxImg,pimg,
                                         rotation_none,rotation)
            print("stats\n\
            inter: {}\n\
            union: {}\n\
            overlap: {}\n".format(np.sum(inter),np.sum(union),overlap))
    return overlaps
