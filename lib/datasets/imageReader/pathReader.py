# --------------------------------------------------------
# CBAL
# Written by Kent Gauen
# --------------------------------------------------------

import os
import os.path as osp
import PIL
import numpy as np
import scipy.sparse
from core.config import cfg
from easydict import EasyDict as edict

class pathReader(object):
    """
    -> reads in raw images @ given image path
    -> reads only one image repository (one file path);
       if you want multi yml files, then use sampleDataset object
    """

    def __init__(self, imgPath, imgExt, is_image_index_flattened = False):
        self._imgPath = imgPath
        self._imgExt = imgExt
        self._is_image_index_flattened = is_image_index_flattened

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        if self._is_image_index_flattened:
            bbox_index = int(index.split('_')[-1])
            image_index = '_'.join(index.split('_')[:-1])
            image_path = os.path.join(self._imgPath,image_index + self._imgExt)
        else:
            image_path = os.path.join(self._imgPath,index + self._imgExt)

        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

