# --------------------------------------------------------
# CBAL
# Written by Kent Gauen
# --------------------------------------------------------

"""Method for loading the datasets by name."""

__sets = []

import os, glob
import numpy as np
from core.config import cfg, cfg_from_file, cfg_from_list
from datasets.repo_imdb import RepoImdb

__sets.append("pascal_voc_2012")
__sets.append("pascal_voc_2007")
__sets.append("pascal_voc")
# __sets.append("imagenet_2014")
# __sets.append("coco_2014")
# __sets.append("coco_2015")
# __sets.append("coco_2015")
# __sets.append("cam2_2017")
# __sets.append("sun_2012")
__sets.append("caltech")
__sets.append("kitti_2013")
__sets.append("inria")

def get_repo_imdb(name):
    """Get an imdb (image database) by name."""
    di = name.split("-")
    if len(di) < 3:
        raise KeyError('Dataset name [{}] is not long enough'.format(name))
    for __set in __sets:
        if di[0] == __set: return RepoImdb(di[0],di[1],di[2])
    raise KeyError('Unknown dataset: {}'.format(name))


def list_imdbs():
    """List all registered imdbs."""
    return __sets
