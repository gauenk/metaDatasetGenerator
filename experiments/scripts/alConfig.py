import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

#
# Global Options
#
__C.SOLVER_STATE = None
__C.AL_INFO_FN = None
__C.AL_MODEL_TYPE = None
__C.NUMBER_OF_PASSES_OF_AL_SUBSET = None
__C.BATCH_SIZE = None
__C.RESTORE_STATE = False
__C.ORIGINAL_MODEL_NAME = None
__C.ORIGINAL_MODEL_TYPE = None
__C.ORIGINAL_MODEL_ITERS = None
__C.ORIGINAL_MODEL_ACCURACY = None
__C.CONFIG_FILE_TO_USE = None
__C.OUTPUT_DIR = "./output/al/"
__C.DATACACHE_STR_MODIFIER = None
__C.SAVE_MOD = 300
__C.AL_SUBSET_VALIDATE_START_ITER = 200
__C.AL_SUBSET_VALIDATE_END_ITER = 4000
__C.AL_SUBSET_VALIDATE_FREQ = 200
__C.ORIGINAL_MODEL_ROOTDIR = "/home/gauenk/Documents/experiments/metaDatasetGenerator/output/classification/mnist/"


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        old_type = type(b[k])
        # the types must match, too; unless old_type is not edict and not None; and new_type is not None
        if old_type is not type(v) and \
        (old_type is edict and old_type is not type(None))\
        and type(v) is not type(None):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        elif v == "None":
            b[k] = None
        else:
            b[k] = v




