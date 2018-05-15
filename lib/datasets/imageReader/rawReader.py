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

class rawReader(object):
    """
    -> reads in raw images @ given image path
    -> reads only one image repository (one file path);
       if you want multi yml files, then use sampleDataset object
    """

    def __init__(self, imgPath, imgExt, image_index):
        self._imgPath = imgPath
        self._imgExt = imgExt
        self._image_index = image_index

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._imgPath,
                                  index + self._imgExt)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

