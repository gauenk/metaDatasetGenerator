import sys,os,pickle,uuid,cv2,glob,csv
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import numpy.random as npr
from core.config import cfg,iconicImagesFileFormat
from core.routingConfig import cfg as cfgRouting
from ntd.hog_svm import plot_confusion_matrix,appendHOGtoRoidb,split_data, scale_data,train_SVM,findMaxRegions, make_confusion_matrix
from datasets.ds_utils import computeTotalAnnosFromAnnoCount

from utils.image_utils import *
from utils.box_utils import *
from utils.imdb_report_utils import *
from utils.mixture_utils import *

def get_roidb(imdb_name):
    from datasets.factory import get_repo_imdb
    from core.train import get_training_roidb
    imdb = get_repo_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    roidb = get_training_roidb(imdb)
    return imdb, roidb

def vis_dets(im, class_names, dets, _idx_, fn=None, thresh=0.5):
    """Draw detected bounding boxes."""
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    if dets is not None:
       for i in range(len(dets)):
           bbox = dets[i, :4]
           ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),bbox[2] - bbox[0],bbox[3] - bbox[1], fill=False,edgecolor='red', linewidth=3.5))
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    uuidStr = str(uuid.uuid4())
    if fn is None:
        plt.savefig("img_{}_{}.png".format(_idx_,uuidStr))
    else:
        plt.savefig(fn.format(_idx_,uuidStr))

def toRadians(angle):
    return (np.pi/180 * angle)

def zeroInTheRegion(coordinate,rows,cols):
    if 0 <= coordinate[0] and coordinate[0] <= rows: coordinate[0] = 0
    if 0 <= coordinate[1] and coordinate[1] <= cols: coordinate[1] = 0

def overflowOnly(coordinate,rows,cols):
    if 0 > coordinate[0]: coordinate[0] = np.abs(coordinate[0])
    elif rows < coordinate[0]: coordinate[0] = rows - coordinate[0]
    if 0 > coordinate[1]: coordinate[1] = np.abs(coordinate[1])
    elif cols < coordinate[1]: coordinate[1] = cols - coordinate[1]

def correctTranslatedIndex(coordinate,rows,cols):
    zeroInTheRegion(coordinate,rows,cols)
    overflowOnly(coordinate,rows,cols)

def getRotationScale(M,rows,cols):
    a = np.array([cols,0,1])
    b = np.array([0,0,1])
    ta = np.matmul(M,a)
    tb = np.matmul(M,b)
    correctTranslatedIndex(ta,rows,cols)
    correctTranslatedIndex(tb,rows,cols)
    scale_a_0 = rows / ( 2. * np.abs(ta[0]) + rows )
    scale_a_1 = rows / ( 2. * np.abs(ta[1]) + rows )
    scale_b_0 = cols / ( 2. * np.abs(tb[0]) + cols )
    scale_b_1 = cols / ( 2. * np.abs(tb[1]) + cols )
    scale_list = [scale_a_0,scale_a_1,scale_b_0,scale_b_1]
    scale = np.min([scale_a_0,scale_a_1,scale_b_0,scale_b_1])
    return scale

def getRotationInfo(angle,cols,rows):
    if cfg._DEBUG.utils.misc: print("cols,rows",cols,rows)
    rotationMat = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1.0)
    scale = getRotationScale(rotationMat,rows,cols)
    if cfg._DEBUG.utils.misc: print("scale: {}".format(scale))
    rotationMat = cv2.getRotationMatrix2D((cols/2,rows/2),angle,scale)
    return rotationMat,scale

def networkForwardBackward(net,inputBlob):
    output = net.forward(**inputBlob)
    net.backward()
    return output,net.layers[-1].diff[0].data

def save_image_of_overlap_bboxes(fn,img_bb1,img_bb2,rot1,rot2):
    img = np.zeros(img_bb1.shape)

    inter = np.bitwise_and(img_bb1.astype(np.bool),img_bb2.astype(np.bool))
    union = np.bitwise_or(img_bb1.astype(np.bool),img_bb2.astype(np.bool))
    overlap = np.sum(inter) / float(np.sum(union))

    img1 = getImageWithBorder(img_bb1,border=255,rotation=rot1)
    img2 = getImageWithBorder(img_bb2,border=255,rotation=rot2)
    img[:,:,0] = npBoolToUint8(inter)[:,:,0] # blue
    img[:,:,1] = img1[:,:,1] # green; the guess
    img[:,:,2] = img2[:,:,2] # red; the groundtruth
    cv2.putText(img,'{:0.2f}'.format(overlap),(0,60),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255,255))
    fp = osp.join(cfg.ROTATE_PATH,fn)
    cv2.imwrite(fp,img)

def save_list_of_bbox_imgs(fn,pimgs):
    img = pimgs[0].astype(np.uint8)
    for pimg in pimgs:
        img = np.bitwise_or(pimg.astype(np.uint8),img)
    fp = osp.join(cfg.ROTATE_PATH,fn)
    cv2.imwrite(fn,img)    

    ADD_NOISE = True
    moves = []
    if ADD_NOISE:
        moves = zip([0,1,-5,3],[0,-1,3,-5])
    base_fn = fn.split(".")[0]
    cols = img.shape[0]
    rows = img.shape[1]
    for idx,m_xy in enumerate(moves):
        xm,ym = m_xy[0],m_xy[1]
        fn = "{}_{}.png".format(base_fn,idx)
        M = np.float32([[1,0,xm],[0,1,ym]])
        dst = cv2.warpAffine(img,M,(cols,rows))
        cv2.imwrite(fn,dst)
        
def npBoolToUint8(a):
    return a.astype(np.uint8)*255

def centerAndScaleBBox(bbox,rotMat,scale):
    if cfg._DEBUG.utils.misc: print("[centerAndScaleBBox] before bbox",bbox)

    cx = np.mean([bbox[0],bbox[2]])
    cy = np.mean([bbox[1],bbox[3]])
    center = np.array([cx,cy,1])
    new_center = np.matmul(rotMat,center)

    x_len = bbox[2] - bbox[0]
    y_len = bbox[3] - bbox[1]

    x1 = int(new_center[0] - 0.5 * scale * x_len)
    x2 = int(new_center[0] + 0.5 * scale * x_len)
    y1 = int(new_center[1] - 0.5 * scale * y_len)
    y2 = int(new_center[1] + 0.5 * scale * y_len)
    new_bbox = [x1,y1,x2,y2]
    if cfg._DEBUG.utils.misc: print("[centerAndScaleBBox] after bbox",new_bbox)

    return new_bbox


def print_net_activiation_data(net,layers_to_print):
    print("-"*50)
    for name,blob in net.blobs.items():
        if name in layers_to_print:
            print("{}: {}".format(name,blob.data.shape))
    print("-"*50)


def createNoisyBox(scale):
    r = npr.rand(4)*scale
    return r - np.mean(r)


# TODO: should we delete this function?
"""
def loadActivityVectors():
    av = {}
    avDir = cfg.GET_SAVE_ACTIVITY_VECTOR_BLOBS_DIR()
    for fn in glob.glob('{}/*.pkl'.format(avDir)):
        start = fn.rfind('/') + 1
        end = fn.rfind('.')
        blob_name = fn[start:end]
        with open(fn,'rb') as f:
            av[blob_name] = pickle.load(f)
    return av
"""

def evaluate_image_detections(roidb,cls_dets):
    """
    used in "alcls_data_layer" and "aim_data_layer"
    we only decide if a bounding box is detected;
    e.g. (TP and FN) only
         NOT (FP or FN)
    """
    # print("[./utils/misc.py: evaluate_image_detections]")
    BBGT = []
    for elem in roidb:
        for idx,cls in enumerate(elem['gt_classes']):
            if cls != 0: continue #cfg.PERSON_INDEX
            BBGT.append(elem['boxes'][idx])
    BBGT = np.array(BBGT)
    # print(BBGT.shape)
    found = np.zeros((BBGT.shape[0])).astype(np.uint8)
    for bb in cls_dets:
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
        found[jmax] = 1
    return found
        
def computeEntropyOfNumpyArray(value):
    cvalue = value.copy()
    cvalue += np.abs(np.min(cvalue))
    cvalue /= cvalue.sum()
    return -(cvalue * np.ma.log(cvalue)/np.log(np.e)).sum()
