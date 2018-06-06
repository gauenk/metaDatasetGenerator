# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------


# metaDatsetGen imports
from core.config import cfg, createFilenameID

# misc imports
import pprint
pp = pprint.PrettyPrinter(indent=4)
import pickle,cv2,uuid
import os.path as osp
import numpy as np

def unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)

def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

def validate_boxes(boxes, width=0, height=0):
    """Check that a set of boxes are valid."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    assert (x1 >= 0).all()
    assert (y1 >= 0).all()
    assert (x2 >= x1).all()
    assert (y2 >= y1).all()
    assert (x2 < width).all()
    assert (y2 < height).all()

def clean_box(box,w,h):
    if box[0] < 0: box[0] = 0
    if box[1] < 0: box[1] = 0

    # if equal; we elect to open the box right 1 pixel
    if box[0] == box[2]: box[2] += 1
    if box[1] == box[3]: box[3] += 1

    if box[0] >= w:
        box[0] = w-1
        box[2] = w

    if box[1] >= h:
        box[1] = h-1
        box[3] = h

    if box[2] > w: box[2] = w
    if box[3] > h: box[3] = h

    return box

def filter_small_boxes(boxes, min_size):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = np.where((w >= min_size) & (h > min_size))[0]
    return keep

def load_mixture_set_single(setID,repetition,size):
    pklName = createFilenameID(setID,str(repetition),str(size)) + ".pkl"
    # write pickle file of the roidb
    if osp.exists(pklName) is True:
        fid = open(pklName,"rb")
        loaded = pickle.load(fid)
        fid.close()

        trainData = loaded['train']
        print_each_size(trainData[0])
        testData = loaded['test']
        print_each_size(testData[0])
    else:
        raise ValueError("{} does not exists".format(pklName))
    return trainData,testData

    
def load_mixture_set(setID,repetition,final_size):

    roidbTr = []
    roidbTe = []
    annoCountsTr = []
    annoCountsTe = []

    datasetSizes = cfg.MIXED_DATASET_SIZES
    if final_size not in datasetSizes:
        print("invalid dataset size")
        print("valid option sizes include:")
        print(datasetSizes)
        raise ValueError("size {} is not in cfg.MIXED_DATASET_SIZES".format(final_size))
    sizeIndex = datasetSizes.index(final_size)
    
    for size in datasetSizes[:sizeIndex+1]:
        # create a file for each dataset size
        pklName = createFilenameID(setID,str(repetition),str(size)) + ".pkl"
        # write pickle file of the roidb
        if osp.exists(pklName) is True:
            fid = open(pklName,"rb")
            loaded = pickle.load(fid)
            train = loaded['train']
            test = loaded['test']
            print(pklName)
            if size == final_size: # only save the last count
                annoCountsTr = train[1]
                annoCountsTe = test[1]
            roidbTr.extend(train[0])
            roidbTe.extend(test[0])
            fid.close()
        else:
            raise ValueError("{} does not exists".format(pklName))
    return {"train":[roidbTr,annoCountsTr],"test":[roidbTe,annoCountsTe]}

def save_mixture_set_single(roidb,annoCount,setID,repetition,size):
    pklName = createFilenameID(setID,str(repetition),str(size)) + ".pkl"
    saveInfo = {"allRoidb":roidb,"annoCounts":annoCount}
    with open(pklName,"wb") as f:
        pickle.dump(saveInfo,f)

def print_each_size(roidb):
    print("="*100)
    sizes = [0 for _ in range(8)]
    for elem in roidb:
        sizes[elem['set']] += len(elem['boxes'])
    print(sizes)

def printPyroidbSetCounts(pyroidb):
    print("="*100)
    import yaml
    fn = "lib/datasets/ymlConfigs/default_dataset_index.yml"
    fn = osp.join(cfg.ROOT_DIR,fn)
    with open(fn,"r") as f:
        setIds = yaml.load(f)

    idsToSet = ["" for _ in range(8)]
    for val,idx in setIds.items():
        idsToSet[idx] = val
    sizes = dict.fromkeys(idsToSet, 0)
    for elem,target in pyroidb:
        sizes[idsToSet[target]] += 1
    pp.pprint(sizes)

def pyroidbTransform_cropImageToBox(inputs,**kwargs):
    im_orig = inputs[0]
    box = inputs[1]
    clean_box(box,kwargs['sample']['width'],kwargs['sample']['height'])
    return cropImageToAnnoRegion(im_orig,box)

def cropImageToAnnoRegion(im_orig,box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    return scaleImage(im_orig[y1:y2, x1:x2])
    
def scaleImage(im_orig):
    target_size = cfg.CROPPED_IMAGE_SIZE 
    x_size,y_size = im_orig.shape[0:2]
    if x_size == 0 or y_size == 0:
        return im_orig
    im_scale_x = float(target_size) / x_size
    im_scale_y = float(target_size) / y_size
    im = cv2.resize(im_orig, (target_size,target_size),
                    interpolation=cv2.INTER_CUBIC)
    return im

def computeTotalAnnosFromAnnoCount(annoCount):
    size = 0
    for cnt in annoCount:
        if cnt is None: continue
        size += cnt
    return size

def compute_size_along_roidb(roidb):
    _roidbSize = []
    if roidb is None:
        raise ValueError("roidb must be loaded before 'compute_size_along_roidb' can be run")
    _roidbSize.append(len(roidb[0]['boxes']))
    for image in roidb[1:]:
        newSize = _roidbSize[-1] + len(image['boxes'])
        _roidbSize.append(newSize)
    return _roidbSize

def roidbSampleImage(sample,annoIndex):
    # load the image
    img = cv2.imread(sample['image'])
    if sample['flipped']:
        img = img[:, ::-1, :]
    return img,sample['set']

def roidbSampleImageAndBox(sample,annoIndex):
    # load the image
    img = cv2.imread(sample['image'])
    if sample['flipped']:
        img = img[:, ::-1, :]
    return [img,sample['boxes'][annoIndex]],sample['set']

def roidbSampleHOG(sample,annoIndex):
    # load the hog
    return sample['hog'][annoIndex],sample['set']

def roidbSampleBox(sample,annoIndex):
    # load the box
    return sample['boxes'][annoIndex],sample['set']

def getFirstElementNotNone(pythonList):
    return next(item for item in pythonList if item is not None)

def addRoidbField(roidb,fieldName,transformFunction):
    firstRoidb = getFirstElementNotNone(roidb)
    if fieldName in firstRoidb.keys():
        print("WARNING: field name [{}] already exists.".format(fieldName))

    totalRoidbs = len(roidb)
    for idx,sample in enumerate(roidb):

        # progress = int(float(idx) / float(totalRoidbs) * 100)
        # print("-"*progress)
        # print("")

        if fieldName in sample.keys() and sample[fieldName] is not None:
            continue
            # sample[fieldName].append(transformFunction(sample))
        sample[fieldName] = transformFunction(sample)

def checkNormalizeSample(sample,annoIndex):
    return ("bbox_noramlized?" in sample.keys() and sample["bbox_noramlized?"][annoIndex] is False) or ("bbox_noramlized?" not in sample.keys())

def initNormalizeSample(sample):
    if "bbox_noramlized?" not in sample.keys():
        sample["bbox_noramlized?"] = [False for _ in range(len(sample['boxes']))]

def updateNormalizeSample(sample,annoIndex):
    sample["bbox_noramlized?"][annoIndex] = True

def pyroidbTransform_normalizeBox(inputs,**kwargs):
    sample = kwargs['sample']
    annoIndex = kwargs['annoIndex']
    if checkNormalizeSample(sample,annoIndex):
        initNormalizeSample(sample)
        inputs = inputs.astype(np.float64)
        inputs = clean_box(inputs,sample['width'],sample['height'])
        inputs[::2] /= sample['width']
        inputs[1::2] /= sample['height']
        if np.any(inputs > 1):
            print(inputs,sample)
        assert np.all(inputs <= 1)
        assert np.all(inputs >= 0)
        updateNormalizeSample(sample,annoIndex)
    return inputs

    
def vis_dets(im, class_names, dets, _idx_, fn=None, thresh=0.5):
    """Draw detected bounding boxes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(len(dets)):
        bbox = dets[i, :4]
        
        
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    if fn is None:
        plt.savefig("img_{}_{}.png".format(_idx_,str(uuid.uuid4())))
    else:
        plt.savefig(fn.format(_idx_,str(uuid.uuid4())))








