# misc imports
from core.configBase import cfgDebug
import os.path as osp
import numpy as np
from numpy import transpose as npt

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

def get_bbox_info(roidb,size):
    areas = np.zeros((size))
    widths = np.zeros((size))
    heights = np.zeros((size))
    actualSize = 0
    idx = 0
    print(size)
    for image in roidb:
        # skipped *flipped* samples
        if image['flipped'] is True: continue
        bbox = image['boxes']
        for box in bbox:
            actualSize += 1
            widths[idx] = box[2] - box[0]
            heights[idx] = box[3] - box[1]
            assert widths[idx] >= 0,"widths[{}] = {}".format(idx,widths[idx])
            assert heights[idx] >= 0
            areas[idx] = widths[idx] * heights[idx]
            idx += 1
    print("actual: {} | theoretical: {}".format(idx,size))
    return areas,widths,heights

def centerAndScaleBBox(bbox,rotMat,scale):
    if cfgDebug.utils.misc: print("[centerAndScaleBBox] before bbox",bbox)

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
    if cfgDebug.utils.misc: print("[centerAndScaleBBox] after bbox",new_bbox)

    return new_bbox


