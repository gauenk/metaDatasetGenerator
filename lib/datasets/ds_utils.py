# metaDatsetGen imports
from utils.base import *
from utils.image_utils import *
from utils.box_utils import *
from utils.mixture_utils import *

from datasets.data_utils.roidb_utils import *
from datasets.data_utils.pyroidb_utils import *
from datasets.data_utils.records_utils import *
from datasets.data_utils.activation_value_utils import *

# misc imports
import uuid,os,sys

def computeTotalAnnosFromAnnoCount(annoCount):
    size = 0
    for cnt in annoCount:
        if cnt is None: continue
        size += cnt
    return size

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
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]),bbox[2] - bbox[0],bbox[3] - bbox[1], fill=False,edgecolor='red', linewidth=3.5))
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    if fn is None:
        plt.savefig("img_{}_{}.png".format(_idx_,str(uuid.uuid4())))
    else:
        plt.savefig(fn.format(_idx_,str(uuid.uuid4())))

def split_and_load_ImdbImages(imdb,records):
    for roidb,image_id in zip(imdb.roidb,imdb.image_set):
        record = records[image_id]

# CALL LIKE: convertFlattenedImageIndextoImageIndex(flattened_image_index,cfg.DATASETS.IS_IMAGE_INDEX_FLATTENED)
def convertFlattenedImageIndextoImageIndex(flattened_image_index,flattened_bool):
    if '_' not in flattened_image_index: return flattened_image_index,0
    if not flattened_bool: return flattened_image_index,0
    bbox_index = flattened_image_index.split('_')[-1]
    image_index = '_'.join(flattened_image_index.split('_')[:-1])
    return image_index,int(bbox_index)

            
