import caffe,os,re
import os.path as osp
from caffe.proto import caffe_pb2
from core.config import cfg,solverPrototxtToYaml,create_snapshot_prefix

def fromSolverprototxtToHelperTrainprototxt(solver_prototxt):
    base = '/'.join(solver_prototxt.split('/')[:-1])
    base += '/train_with_data_input.prototxt'
    return base

def getAllNetLayers(net):
    layerTypeList = []
    layerIndex = 0
    for layer,layerName in zip(net.layers,net._layer_names):
        addLayer = [layer,layerName,layerIndex]
        layerTypeList.append(addLayer)
        layerIndex += 1
    return layerTypeList

def getLayerDictFromNet(net):
    layer_dict = {}
    for layer,layerName in zip(net.layers,net._layer_names):
        layer_dict[layerName] = layer
    return layer_dict
    
def findLayerType(net,layerTypeStr,pythonTypeStr=None):
    layerTypeList = []
    layerIndex = 0
    for layer,layerName in zip(net.layers,net._layer_names):
        addLayer = [layer,layerName,layerIndex]
        layerIndex+=1
        if layer.type == layerTypeStr:
            if pythonTypeStr is None:
                layerTypeList.append(addLayer)
            elif pythonTypeStr == layer.name:
                layerTypeList.append(addLayer)
    return layerTypeList

def findLayerName(net,inputLayerName,regex=False):
    layerTypeList = []
    layerIndex = 0
    for layer,layerName in zip(net.layers,net._layer_names):
        addLayer = [layer,layerName,layerIndex]
        layerIndex+=1
        if regex is False:
            if layerName == inputLayerName:
                layerTypeList.append(addLayer)
        else:
            matches = re.match(inputLayerName,layerName)
            if matches:
                name = matches.groupdict()['name']
                addLayer[1] = name
                layerTypeList.append(addLayer)
    return layerTypeList


def setNetForWarpAffineLayerType(net,solver_prototxt):
    train_helper_prototxt = fromSolverprototxtToHelperTrainprototxt(solver_prototxt)
    layerTypeList = findLayerType(net,'Python','WarpAffineLayer')
    if len(layerTypeList) == 0:
        return 
    net_copy = caffe.Net(train_helper_prototxt,caffe.TRAIN)
    net_copy.share_with(net)
    for layer,layerName,layerIndex in layerTypeList:
        layer.train_mode = True
        net_copy.layers[layerIndex].batch_size = layer.search.step_number
        layer.set_net(net,net_copy,layerName)
    print("setNetForWarpAffineLayerType successful")

def initializeHighwayLayerBiases(net):
    pass

def checkLayerType(net,layerTypeStr,pythonTypeStr=None):
    layerTypeList = findLayerType(net,layerTypeStr,pythonTypeStr=pythonTypeStr)
    if len(layerTypeList) == 0:
        return False
    else:
        return True

def checkLayerName(net,inputLayerName,regex=False):
    layerNameList = findLayerName(net,inputLayerName,regex=False)
    if len(layerNameList) == 0:
        return False
    else:
        return True


def loadNetFromLayers(layerList):
    raise NotImplemented("[loadNetFromLayers] not implemented...")

def initializeWarpAffineLayers_version1(net,warp_affine):
    """
    1. copy the "warp_" prefix layers into a new net
    2. load the caffemodel into the net net
    3. copy the loaded weights into the original model
    """
    if warp_affine is None:
        return
    # 1. find layers with "warp_" prefix in the layer name
    load_net = caffe_pb2.NetParameter()
    load_net.name = "warp_affine_loading_helper"

    layers = findLayerName(net,"warp_*",regex=True)
    for layer,layerName,layerIndex in layers:
        new_load_layer = caffe.Layer(layer.param)
    print("HI")
    exit()

def initializeWarpAffineLayers_version2(net,warp_affine_net,warp_affine_def):
    if warp_affine_net is None or warp_affine_def is None:
        return
    warp_layername_regex = "warp_(?P<name>.*)"
    # 1. find layers with "warp_" prefix in the layer name
    load_net = caffe.Net(warp_affine_def,warp_affine_net,caffe.TEST)
    layers_for_filling = getLayerDictFromNet(load_net)
    layers_to_fill = findLayerName(net,warp_layername_regex,regex=True)
    for layer,layerName,layerIndex in layers_to_fill:
        do_we_fill = layerName in layers_for_filling.keys()
        if do_we_fill:
            fill_layer = layers_for_filling[layerName]
            copyLayerWeights(layer,fill_layer)

def copyLayerWeights(layer,fill_layer):
    for blob_to_fill,filling_blob in zip(layer.blobs,fill_layer.blobs):
        blob_to_fill.data[...] = filling_blob.data

def initializeNetworkWeights(net,solver_state,warp_affine_net,warp_affine_def):
    if solver_state is None and checkLayerType(net,'highway'):
        initializeHighwayLayerBiases(net)
    if checkLayerType(net,'Python','WarpAffineLayer'):
        initializeWarpAffineLayers_version2(net,warp_affine_net,warp_affine_def)

def insertInfixBeforeDecimal(oMsg,infix):
    splitList = oMsg.split(".")
    assert len(splitList) in [2,3], "splitList is length not 2 or 3: {}".format(len(splitList))
    splitList[-2] += infix
    return '.'.join(splitList)
    
def writeSolverToFile(fn,ymlContent):
    ymlKeys = ymlContent.keys()
    useQuotesList = ["lr_policy","train_net","snapshot_prefix","type"]
    with open(fn, 'w') as f:
        for key in ymlKeys:
            val = ymlContent[key]
            useQuotes = key in useQuotesList
            if useQuotes:
                f.write("{}: \"{}\"\n".format(key,val))
            else:
                f.write("{}: {}\n".format(key,val))                

def addFullPathToSnapshotPrefix(solverYaml,outputDir):
    snapshotPrefix = solverYaml['snapshot_prefix']
    if "/home/" not in solverYaml['snapshot_prefix']:
        finalSubstr = solverYaml['snapshot_prefix']
        snapshotPrefix = osp.join(outputDir,solverYaml['snapshot_prefix'])
        # add infix to ymlData
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        solverYaml['snapshot_prefix'] = snapshotPrefix + infix
        print("new snapshot_prefix: {}".format(snapshotPrefix))

def resetSnapshotPrefix(solverYaml,new_snapshot_prefix):
    solverYaml['snapshot_prefix'] = new_snapshot_prefix

def mangleSolverPrototxt(solverPrototxt,outputDir,modelInfo,recreate_snapshot_name=True,append_string=None):
    print("Mangling solverprototxt {}".format(solverPrototxt))
    infix = "_generatedByTrainpy"
    newSolverPrototxtFilename = insertInfixBeforeDecimal(solverPrototxt,infix)
    solverYaml = solverPrototxtToYaml(solverPrototxt)
    print("writing new solver_prototxt @ {}".format(newSolverPrototxtFilename))
    
    # create snapshot prefix name
    if recreate_snapshot_name:
        new_snapshot_prefix = create_snapshot_prefix(modelInfo)
        if append_string:
            new_snapshot_prefix += "_{}".format(append_string)
        resetSnapshotPrefix(solverYaml,new_snapshot_prefix)
    
    # add full path
    addFullPathToSnapshotPrefix(solverYaml,outputDir)
    
    # write the new solver_prototxt
    writeSolverToFile(newSolverPrototxtFilename,solverYaml)
    print("added full path to snapshot_prefix")
    print("snapshot_prefix is [{}]".format(solverYaml['snapshot_prefix']))

    return newSolverPrototxtFilename


