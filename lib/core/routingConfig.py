import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict
from core.configUtils import _merge_a_into_b

__C = edict()
# Consumers can get config by:
#   from core.config import cfg
cfg = __C


cfg.expName = 'default'
cfg.loadOnlyClsStr = None # shortcut for development; only load the class name speficied; e.g. "cat" for cifar_10
cfg.verbose = False # only use this if we want to print a lot.

cfg.netInfo = edict()
cfg.netInfo.modelName = None
cfg.netInfo.modelType = None
cfg.netInfo.trainDataset = edict()
cfg.netInfo.trainDataset.name = None
cfg.netInfo.trainDataset.split = None
cfg.netInfo.trainDataset.config = None

cfg.routingAnalysisInfo = edict()
#cfg.routingAnalysisInfo.layers = ["conv1","conv2","ip1","cls_score"]
cfg.routingAnalysisInfo.layers = ["conv1"]
cfg.routingAnalysisInfo.clsReferenceRouteStr = 'Pos'

# cfg.routingAnalysisInfo.densityEstimationType = 'cluster'
# cfg.routingAnalysisInfo.densityEstimationClusterType = 'kmeans'
# cfg.routingAnalysisInfo.densityEstimationTypeConfig = None
# cfg.routingAnalysisInfo.densityEstimationTypeConfigFilename = None
# cfg.routingAnalysisInfo.densityEstimationTypeConfigByClass = True # always true currently; we can specify how we want classes grouped together in [routingStatisticsByClass] (e.g. not just by class )

cfg.routingAnalysisInfo.comboType = 'all'
cfg.routingAnalysisInfo.comboInfo = None #[None _ for combo in comboList]
cfg.routingAnalysisInfo.routeFunction = None # generated from ..."str"
cfg.routingAnalysisInfo.routeDifference = None # generated from ..."str"
cfg.routingAnalysisInfo.routeFunctionStr = None # function type to creating a route from activ.
cfg.routingAnalysisInfo.routeDifferenceStr = None # function type to take the difference between two routes

cfg.routingAnalysisInfo.densityEstimation = edict()
cfg.routingAnalysisInfo.densityEstimation.typeStr = 'cluster'
cfg.routingAnalysisInfo.densityEstimation.clusterTypeStr = 'kmeans'
cfg.routingAnalysisInfo.densityEstimation.typeConfig = None
cfg.routingAnalysisInfo.densityEstimation.typeConfigFilename = None
cfg.routingAnalysisInfo.densityEstimation.allClassesSameParameters = True
cfg.routingAnalysisInfo.densityEstimation.classSeparate = True # always true currently; we can specify how we want classes grouped together in [routingStatisticsByClass] (e.g. not just by class )

# training dataset for routing analysis
cfg.routingAnalysisInfo.train = edict()
cfg.routingAnalysisInfo.train.name = None
cfg.routingAnalysisInfo.train.split = None
cfg.routingAnalysisInfo.train.config = None
cfg.routingAnalysisInfo.train.size = None
# testing dataset for routing analysis
cfg.routingAnalysisInfo.test = edict()
cfg.routingAnalysisInfo.test.name = None
cfg.routingAnalysisInfo.test.split = None
cfg.routingAnalysisInfo.test.config = None
cfg.routingAnalysisInfo.test.size = None


#
# classification experiment info
#

cfg.clsExperimentInfo = edict()
cfg.clsExperimentInfo.clsModelType = 'Svm'
# reference routing information
cfg.clsExperimentInfo.referenceRoute = edict()
cfg.clsExperimentInfo.referenceRoute.referenceName = None
cfg.clsExperimentInfo.referenceRoute.indexWeightStr = None
cfg.clsExperimentInfo.referenceRoute.dataset = edict()
cfg.clsExperimentInfo.referenceRoute.dataset.name = None
cfg.clsExperimentInfo.referenceRoute.dataset.split = None
cfg.clsExperimentInfo.referenceRoute.dataset.config = None

#
# if we use clustering for density esimation, these are the configs for each method
#
cfg.kmeans = edict()
cfg.kmeans.nClusters = 100

cfg.dbscan = edict()
cfg.dbscan.eps = 12500
cfg.dbscan.minSamples = 2

def cfg_from_file(filename):
    """Load a config file (NO merging) it into the default options."""
    yaml_cfg = loadYmlFile(filename)
    _merge_a_into_b(yaml_cfg, __C)

def loadYmlFile(filename):
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    return yaml_cfg

def initRouteConfig(densityEstimationConfigFile):
    initDensityEsimationParameterConfig(densityEstimationConfigFile)
    initComboInfo()

def initDensityEsimationParameterConfig(filename):
    cfg.routingAnalysisInfo.densityEstimation.typeConfigFilename = osp.splitext(osp.basename(filename))[0]
    cfg.routingAnalysisInfo.densityEstimation.typeConfig = loadYmlFile(filename)

def initComboInfo():
    layers = cfg.routingAnalysisInfo.layers
    comboType = cfg.routingAnalysisInfo.comboType
    print(layers)
    if comboType == 'all':
        comboInfo = createCombinationForDensityEstimationAll(layers)
    elif comboType == 'pair':
        comboInfo = createCombinationForDensityEstimationPairwise(layers)
    elif comboType == 'sep':
        comboInfo = createCombinationForDensityEstimationSeparate(layers)
    print("combo type is ~{}~.... nice (^_^)".format(comboType))
    cfg.routingAnalysisInfo.comboInfo = comboInfo

def createRecordsPath(modelType,imdbName,netName):
    recordsPath = "./output/{modelType}/tp_fn_records/{imdbName}/records_cls_{netName}.pkl"\
                   .format(modelType=modelType,imdbName=imdbName,netName=netName)
    return recordsPath
    
def createActivityVectorPath(modelType,imdbName,netName):
    recordsPath = "./output/{modelType}/tp_fn_records/{imdbName}/records_cls_{netName}.pkl"\
                   .format(modelType=modelType,imdbName=imdbName,netName=netName)
    return recordsPath

def createDensityEstimationCacheStrID(comboID,recordIncludeTypeStr,imdb,clsName):
    dsName,dsSplit,dsConfig = imdbToVariableStrings(imdb)
    allClassesSameParameters = cfg.routingAnalysisInfo.densityEstimation.allClassesSameParameters
    if allClassesSameParameters:
        comboParameters = cfg.routingAnalysisInfo.comboInfo[comboID]
    else:
        comboParameters = cfg.routingAnalysisInfo.comboInfo[comboID][clsName]
    typeStr = cfg.routingAnalysisInfo.densityEstimation.typeStr
    comboStr = createComboInfoStr(comboID,comboParameters,typeStr,clsName)
    return '{comboStr}_{dsName}-{dsSplit}-{dsConfig}_{clsName}_{recordIncludeTypeStr}'.\
        format(comboStr=comboStr,dsName=dsName,dsSplit=dsSplit,dsConfig=dsConfig,
               clsName=clsName,recordIncludeTypeStr=recordIncludeTypeStr)

def createClassificationExperimentCacheName():
    dirPath = "data/routing_cache/{modelName}".format(modelName=cfg.netInfo.modelName)
    if not osp.exists(dirPath): os.makedirs(dirPath)
    fn = "{dirPath}/clsCache.pkl".format(dirPath=dirPath)
    return fn

def imdbFromDatasetDict(datasetDict):
    return "{}-{}-{}".format(datasetDict.name,datasetDict.split,datasetDict.config)

def imdbToVariableStrings(imdb):
    dsName = imdb.name
    dsSplit = imdb._image_set
    dsConfig = 'default'
    return dsName,dsSplit,dsConfig

""" 

conflict currently

we have density estimation parameters for both 'layerNames' and 'classNames'

we can only do one or have some set of rules for combining them...

currently:
-> save name is off of global configuration
-> actual density methods depend on "comboIDs" and "classNames"

"""

def getDensityEstimationHyperparameters(typeStr = None, clusterTypeStr = None):
    if typeStr is None:
        typeStr = cfg.routingAnalysisInfo.densityEstimation.typeStr
    if clusterTypeStr is None:
        clusterTypeStr = cfg.routingAnalysisInfo.densityEstimation.clusterTypeStr
    params = None
    # add more types of density information here
    if cfg.routingAnalysisInfo.densityEstimation.allClassesSameParameters:
        if typeStr == 'cluster':
            if clusterTypeStr == 'kmeans':
                print("--- using some kmeans cluster. aaand that's how the news goes. ----")
                params = templateKMeans(cfg.kmeans.nClusters)
            elif clusterTypeStr == 'dbscan':
                params = templateDBSCAN(cfg.dbscan.minSamples,cfg.dbscan.eps)
        elif typeStr == 'aNewTypeOfDensityEstimation':
            print("A new type of density estimation goes here. BALLERZ")
        else:
            print("--- using the plain old arithmetic average. aaand that's how the news goes. ----")
    else:
        paramsByClass = cfg.routingAnalysisInfo.densityEstimation.typeConfig
        flattenParams = {}
        for name,classParams in params.items():
            flattenParams[name] = paramsByClassclassParams[typeStr][clusterTypeStr]
        params = flattenParams
    return params

def createCombinationForDensityEstimationAll(layers):
    densityEsimtationType = cfg.routingAnalysisInfo.densityEstimation.typeStr
    # currently each layer uses the same stuff
    combos = []
    keyStr = '-'.join(layers)
    densityEstimationInfo = {"method": "cluster"}
    densityEstimationInfo[densityEsimtationType] = getDensityEstimationHyperparameters()
    combos = {keyStr:densityEstimationInfo}
    return combos
    
def createCombinationForDensityEstimationPairwise(layers):
    # currently each layer uses the same stuff
    typeStr = cfg.routingAnalysisInfo.densityEstimation.typeStr
    deTypeConfig = cfg.routingAnalysisInfo.densityEstimation.typeConfig
    clusterTypeStr = cfg.routingAnalysisInfo.densityEstimation.clusterTypeStr
    combos = {}
    for idx,layerNameA in enumerate(layers):
        for jdx,layerNameB in enumerate(layers):
            if idx > jdx: continue
            densityEstimationInfo = {"method": typeStr}
            densityEstimationInfo[typeStr] = getDensityEstimationHyperparameters()
            keyStr = "{}-{}".format(layerNameA,layerNameB)
            if layerNameA == layerNameB: keyStr = "{}".format(layerNameA)
            combos[keyStr] = densityEstimationInfo
    return combos

def createCombinationForDensityEstimationSeparate(layers):
    # currently each layer uses the same stuff
    densityEsimtationType = cfg.routingAnalysisInfo.densityEstimation.typeStr
    combos = {}
    for layer in layers:
        densityEstimationInfo = {"method": "cluster"}
        densityEstimationInfo[densityEsimtationType] = getDensityEstimationHyperparameters()
        keyStr = layer
        combos[keyStr] = densityEstimationInfo
    return combos

def templateKMeans(nClusters):
    kmeans = {}
    kmeans["type"] = "kmeans"
    kmeans["parameters"] = {}
    kmeans["parameters"]['nCluster'] = nClusters
    return kmeans

def templateDBSCAN(minSamples,eps):
    dbscan = {}
    dbscan["type"] = "dbscan"
    dbscan["parameters"] = {}
    dbscan["parameters"]['minSamples'] = minSamples
    dbscan["parameters"]['eps'] = eps
    return dbscan

def getResultsBaseFilenameRouting():
    fn = "{}_{}_{}_{}".format(cfg.routingAnalysisInfo.densityEstimation.typeStr,
                                 cfg.routingAnalysisInfo.densityEstimation.clusterTypeStr,
                                 cfg.routingAnalysisInfo.densityEstimation.typeConfigFilename,
                                 cfg.routingAnalysisInfo.densityEstimation.classSeparate)
    # if cfg.clusterParams is not None and cfg.clusterType is not None:
    #     fn += "_{}".format(cfg.clusterType)
    #     for key,value in cfg.clusterParams.items():
    #         fn += "_{key}{value}".format(key=key,value=value)
    return fn

def getResultsDirectory():
    fn = "./output/routing/{modelName}".format(modelName=cfg.netInfo.modelName)
    return fn

def getClassificationExperimentResultsTxtFilenameRouting(dsName,dsSplitWRTRoute,dsConfigWRTRoute,dsSplitOg,dsConfigOg):
    resultDirName = getResultsBaseFilenameRouting()
    firstDir = "./output/routing/{modelName}".format(modelName=cfg.netInfo.modelName)
    uuidStr = str(uuid.uuid4())
    resultsFn = "{}/results_{}".format(getResultsDirectory(),uuidStr)


    secondDir = "{resultDirName}/{dsName}".format(resultDirName=resultDirName,dsName=dsName)
    thirdDir = "{dsSplitWRTRoute}-{dsConfigWRTRoute}_{dsSplitOg}-{dsConfigOg}_{comboType}".\
               format(dsSplitWRTRoute=dsSplitWRTRoute,dsConfigWRTRoute=dsConfigWRTRoute,
        dsSplitOg=dsSplitOg,dsConfigOg=dsConfigOg,comboType=cfg.routingAnalysisInfo.comboType)
    fullPath = "{firstDir}/{secondDir}/{thirdDir}/".\
               format(firstDir=firstDir,secondDir=secondDir,thirdDir=thirdDir)
    if not osp.exists(fullPath): os.makedirs(fullPath)
    fn = fullPath
    typeStr = cfg.routingAnalysisInfo.densityEstimation.typeStr
    clsExpClassName = cfg.loadOnlyClsStr
    for comboID,comboParameters in cfg.routingAnalysisInfo.comboInfo.items():
        fn += comboID
        #fn += createComboInfoStr(comboID,comboParameters,typeStr,clsExpClassName,short=True)
    refRouteInfo = cfg.clsExperimentInfo.referenceRoute
    fn += "_{}_{}".format(refRouteInfo.referenceName,imdbFromDatasetDict(refRouteInfo.dataset))
    return fn + ".txt"

def createComboInfoStr(comboID,comboParameters,densityEsimtationType,clsName,short=False):
    typeStr = cfg.routingAnalysisInfo.densityEstimation.typeStr
    clusterTypeStr = cfg.routingAnalysisInfo.densityEstimation.clusterTypeStr
    comboStr = "{}+{}+{}".format(comboID,typeStr,clusterTypeStr)
    if not short:
        for key,value in comboParameters[typeStr].items():
            if type(value) is edict:
                comboStr += "_{}".format(key)
                for k,v in value.items():
                    comboStr += "+{}{}".format(k,v)
            else:
                comboStr += "_{}{}".format(key,value)
    return comboStr

def checkConfigEquality(validConfig,proposedConfig):
    """
    check if the input config edict is the same
    as the current config edict
    """
    for key,validValue in validConfig.items(): # iterate through the "truth"
        proposedValue = proposedConfig[key]
        if type(validValue) is edict or type(validValue) is dict:
            isValid = checkConfigEquality(validValue,proposedValue)
            if not isValid: return False
            continue
        if proposedValue != validValue: return False
    return True

def unpackClassificationInformation(info):
    avDict = info['avDict']
    records = info['records']
    layerOrder = info['layerOrder']
    referenceRoute = info['referenceRoute']
    datasetSize = info['datasetSize']
    clsName = info['clsName']
    imdb = info['imdb']
    return avDict,records,layerOrder,referenceRoute,datasetSize,clsName,imdb

def packClassificationInformation(avDict,records,layerOrder,referenceRoute,datasetSize,clsName,imdb):
    info = {}
    info['avDict'] = avDict
    info['records'] = records
    info['layerOrder'] = layerOrder
    info['referenceRoute'] = referenceRoute
    info['datasetSize'] = datasetSize
    info['clsName'] = clsName
    info['imdb'] = imdb
    return info

"""
- network information 
   - model name
   - model architecture
   - dataset used for parameter estimation
- routing analysis info
   - dataset name & split for training *routing analysis*
   - dataset name & split for testing *routing analysis*
   - layers used in the routing analysis
- combination of the layers for density estimation (kmeans, dbscan, etc)
   - split
   - pairs
   - all
   for each combination, density estimation information 
      - type of density estimation
      - parameters for density estimation
- reference route information
   - type (all,pos,neg)
   - dataset used for reference route
   - index weight
- route function information
   - route creation function
   - route difference function
"""



