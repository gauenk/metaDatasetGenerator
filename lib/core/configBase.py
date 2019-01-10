import os,re,yaml
import os.path as osp
from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by:
#   from core.config import cfg
cfg = __C

# output for recoding the TP and FN of a model
__C.TP_FN_RECORDS_PATH = "./output/{}/tp_fn_records/".format("faster_rcnn")
__C.TP_FN_RECORDS_WITH_IMAGESET = True



def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    merge_a_into_b(yaml_cfg, __C)
    load_tp_fn_record_path()

def load_tp_fn_record_path():
    __C.TP_FN_RECORDS_PATH = "./output/{:s}/tp_fn_records/"
    if type(__C.DATASETS.MODEL) is str:
        __C.TP_FN_RECORDS_PATH = __C.TP_FN_RECORDS_PATH.format(__C.DATASETS.MODEL)
    else:
        __C.TP_FN_RECORDS_PATH = __C.TP_FN_RECORDS_PATH.format("faster_rcnn")
    return __C.TP_FN_RECORDS_PATH

def merge_a_into_b(a,b):
    return _merge_a_into_b(a, b)

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


# misc functions... not sure where they should live


def create_snapshot_prefix(modelInfo):
    train_set_snapshot = modelInfo.train_set.replace('_','-')
    name = "{}_{}_{}".format(train_set_snapshot,modelInfo.architecture,modelInfo.optim)
    
    #if cfg.TRAIN.IMAGE_NOISE is not None or cfg.TRAIN.IMAGE_NOISE is not 0:
    if modelInfo.image_noise is not False:
        name += "_yesImageNoise{}".format(modelInfo.image_noise)
    else:
        name += '_noImageNoise'
    
    #if cfg.PRUNE_NET is not None or cfg.PRUNE_NET is not 0:
    if modelInfo.prune is not False:
        name += "_yesPrune{}".format(modelInfo.prune)
    else:
        name += '_noPrune'
        
    #if cfg.DATASET_AUGMENTATION.BOOL:
    if modelInfo.dataset_augmentation is not False:
        name += '_yesDsAug{}'.format(modelInfo.dataset_augmentation)
    else:
        name += '_noDsAug'
    
    #if cfg.DATASETS.CLASS_FILTER:
    if modelInfo.classFilter is not False:
        name += '_yesClassFilter{}'.format(modelInfo.classFilter)
    else:
        name += '_noClassFilter'

    return name

def prototxtToYaml(prototxt):
    import yaml
    from easydict import EasyDict as edict
    print(prototxt)
    with open(prototxt, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    return yaml_cfg 

def getFieldFromSolverptotxt(prototxt,field_name):
    solverYaml = prototxtToYaml(prototxt)
    if field_name in solverYaml:
        return solverYaml[field_name]
    elif field_name is 'type': return 'sgd'
    else: return None

def solverPrototxtInfoDict(solverPrototxt):
    regex = r'.*models/(?P<ds_name>[a-zA-Z0-9_]+)/(?P<arch>[a-zA-Z0-9_]+).*solver\.prototxt'
    _result = re.match(regex,solverPrototxt)
    if _result is None:
        print("no match for solverprototxt regex... quitting")
        exit()
    result = _result.groupdict()
    result['optim'] = getFieldFromSolverptotxt(solverPrototxt,'type')
    return result

def buildNetFromPrototxt(prototxt):
    from caffe.proto import caffe_pb2
    import google.protobuf.text_format as txtf
    net = caffe_pb2.NetParameter()
    with open(prototxt,'r') as f:
        s = f.read()
        txtf.Merge(s,net)
    return net
    
def appendPrototxtWithLayer(layer,prototxt):
    with open(prototxt,'a+') as f:
        f.write('layer {\n')
        f.write(str(layer))
        f.write('}\n')

def appendPrototxtWithNet(net,prototxt):
    with open(prototxt,'a+') as f:
        f.write(str(net))

def writeNetFromProto(net,new_prototxt):
    with open(new_prototxt,'w') as f:
        f.write(str(net))

def checkListEqualityWithOrder(list_a,list_b):
    print(list_a)
    if len(list_a) == 0 and len(list_b) == 0:
        return True
    if len(list_a) != len(list_b):
        return False
    for item_a in list_a:
        for item_b in list_b:        
            if type(item_a) is list and type(item_b) is list:
                if not checkListEqualityWithOrder(item_a,item_b):
                    return False
            elif type(item_a) is edict and type(item_b) is edict:
                if checkEdictEquality(item_a,item_b):
                    return False
            elif type(item_a) is dict and type(item_b) is dict:
                if checkEdictEquality(item_a,item_b):
                    return False
            elif item_a != item_b:
                return False
    return True


# def check_list_equal_any(list_a,list_b):
#     # assumes the types are aligned
#     if len(list_a) == 0 or len(list_b) == 0:
#         return False
#     for item_a in list_a:
#         if item_b in list_b:
#             if type(item_a) is list:
#                 if check_list_equal_any(item_a,item_b):
#                     return True
#             if type(item_a) is 
#             if item_a == item_b:
#                 return True
#     return False

def any_true_in_list(boolList):
    anyTrue = False
    for boolValue in boolList:
        if boolValue is True:
            anyTrue = True
    anyTrue = boolList

def checkListEqualityWithOrderIgnored(list_a,list_b):
    # all the elements in list_a are somewhere in list_b
    if len(list_a) == 0 and len(list_b) == 0:
        return True
    if len(list_a) != len(list_b):
        return False
    for item_a in list_a:
        boolList = []
        for item_b in list_b:        
            if type(item_a) is list and type(item_b) is list:
                if checkListEqualityWithOrderIgnored(item_a,item_b):
                    boolList.append(True)
                else:
                    boolList.append(False)
            elif type(item_a) is edict and type(item_b) is edict:
                if checkEdictEquality(item_a,item_b):
                    boolList.append(True)
                else:
                    boolList.append(False)
            elif type(item_a) is dict and type(item_b) is dict:
                if checkEdictEquality(item_a,item_b):
                    boolList.append(True)
                else:
                    boolList.append(False)
            elif item_a != item_b:
                boolList.append(True)
            else:
                boolList.append(True)
        if any_true_in_list(boolList):
            return False
    return True


def checkEdictEquality(validConfig,proposedConfig):
    """
    check if the input config edict is the same
    as the current config edict
    """
    for key,validValue in validConfig.items(): # iterate through the "truth"
        if key not in proposedConfig.keys():
            return False
        proposedValue = proposedConfig[key]
        if type(validValue) is list:
            isValid = checkListEqualityWithOrderIgnored(validValue,proposedValue)
            if not isValid:
                return False
            continue
        if type(validValue) is edict or type(validValue) is dict:
            isValid = checkEdictEquality(validValue,proposedValue)
            if not isValid:
                return False
            continue
        if proposedValue != validValue:
            return False
    return True
