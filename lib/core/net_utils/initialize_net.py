import caffe,os
import os.path as osp
from core.config import cfg,prototxtToYaml,create_snapshot_prefix

def fromSolverprototxtToHelperTrainprototxt(solver_prototxt):
    base = '/'.join(solver_prototxt.split('/')[:-1])
    base += '/train_with_data_input.prototxt'
    return base

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

def checkLayerType(net,layerTypeStr):
    return True

def checkLayerName(net,layerNameStr):
    return True

def initializeNetworkWeights(net):
    if checkLayerType(net,'highway'):
        initializeHighwayLayerBiases(net)

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

def mangleSolverPrototxt(solverPrototxt,outputDir):
    print("Mangling solverprototxt {}".format(solverPrototxt))
    infix = "_generatedByTrainpy"
    newSolverPrototxtFilename = insertInfixBeforeDecimal(solverPrototxt,infix)
    solverYaml = prototxtToYaml(solverPrototxt)
    print("writing new solver_prototxt @ {}".format(newSolverPrototxtFilename))
    
    # create snapshot prefix name
    if cfg.TRAIN.RECREATE_SNAPSHOT_NAME:
        new_snapshot_prefix = create_snapshot_prefix(cfg.modelInfo)
        resetSnapshotPrefix(solverYaml,new_snapshot_prefix)
    
    # add full path
    addFullPathToSnapshotPrefix(solverYaml,outputDir)
    
    # write the new solver_prototxt
    writeSolverToFile(newSolverPrototxtFilename,solverYaml)
    print("added full path to snapshot_prefix")
    print("snapshot_prefix is [{}]".format(solverYaml['snapshot_prefix']))

    return newSolverPrototxtFilename


