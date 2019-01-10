import numpy as np
import yaml
from utils.image_utils import cropImageToAnnoRegion

def pyroidbTransform_cropImageToBox(inputs,**kwargs):
    im_orig = inputs[0]
    box = inputs[1]
    clean_box(box,kwargs['sample']['width'],kwargs['sample']['height'])
    return cropImageToAnnoRegion(im_orig,box)

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
            print("inputs greater than 1")
        assert np.all(inputs <= 1)
        assert np.all(inputs >= 0)
        updateNormalizeSample(sample,annoIndex)
    return inputs

def printPyroidbSetCounts(pyroidb,rootDir):
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

def checkNormalizeSample(sample,annoIndex):
    return ("bbox_noramlized?" in sample.keys() and sample["bbox_noramlized?"][annoIndex] is False) or ("bbox_noramlized?" not in sample.keys())

def initNormalizeSample(sample):
    if "bbox_noramlized?" not in sample.keys():
        sample["bbox_noramlized?"] = [False for _ in range(len(sample['boxes']))]

def updateNormalizeSample(sample,annoIndex):
    sample["bbox_noramlized?"][annoIndex] = True


