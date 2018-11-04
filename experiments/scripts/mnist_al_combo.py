#!/usr/bin/env python2
import os,sys,re,yaml,subprocess,pickle,uuid
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
from alConfig import cfg,cfg_from_file
import argparse
import pprint
import time, os, sys

outputDir = "./output/al/"
# NO SOLVERSTATE SINCE EACH MODEL IS TRAINED FROM SCRATCH
# PROPORTIONAL COMPUTATION IS ADDED AS THE SIZE OF THE DATASET IN INCREASE
# e.x. original dataset size == 3000  || original # of iters == 1000
#      new dataset size == 3500 || new # of iters == 1166

trainCommandTemplate_noSolverState = "./tools/train_net.py --iters {} --imdb mnist-{}-default --cfg ./experiments/cfgs/cls_mnist_al.yml --solver ./models/mnist/{}/solver.prototxt --new_path_to_imageset {} --snapshot_infix {} --cacheStrModifier {}"
trainCommandTemplate_yesSolverState = "./tools/train_net.py --iters {} --imdb mnist-{}-default --cfg ./experiments/cfgs/cls_mnist_al.yml --solver ./models/mnist/{}/solver.prototxt --solver_state ./output/classification/mnist/{}_iter_{}.solverstate --new_path_to_imageset {} --snapshot_infix {} --cacheStrModifier {}"

# trainCommandTemplate = "./tools/train_net.py --iters {} --imdb mnist-{}-default --cfg ./experiments/cfgs/cls_mnist_al.yml --solver ./models/mnist/{}/solver.prototxt --new_path_to_imageset {} --snapshot_infix {}"
testCommandTemplate = "./tools/test_net.py --imdb mnist-test-default --cfg ./experiments/cfgs/cls_mnist_al.yml --def ./models/mnist/{}/test.prototxt --net ./output/classification_al_subset/mnist/mnist_al_{}_{}_iter_{}.caffemodel"

trainOriginalCommandTemplate = "./tools/train_net.py --iters {} --imdb mnist-short_train-default --cfg ./experiments/cfgs/cls_mnist_al.yml --solver ./models/mnist/{}/solver.prototxt"
testOriginalModelCommandTemplate = "./tools/test_net.py --imdb mnist-test-default --cfg ./experiments/cfgs/cls_mnist_al.yml --def ./models/mnist/{}/test.prototxt --net {}"

def dbprint(item):
    print("DEBUG_PRINT")
    print(item)
    
def getAlInfo(fn):
    with open(fn, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    return yaml_cfg

def assignAccuracy(alIds,alSet,acc,testIter,alSubsetSize):
    alSubset = getAlSubsetInfo(alSet)[:alSubsetSize]
    for alId in alSubset:
        if alIds[alId] is None: alIds[alId] = {}
        if testIter not in alIds[alId].keys(): alIds[alId][testIter] = []
        alIds[alId][testIter].append(acc)
    
def getIdList(imageSetPath):
    with open(imageSetPath,"r") as f:
        lines = f.readlines()
    ids = [int(line.strip()) for line in lines]
    return ids
    
def runCommandProcess(setCommand):
    modelProc = subprocess.Popen(setCommand.split(' '),stdout=subprocess.PIPE)
    output_b,isSuccess = modelProc.communicate()
    assert isSuccess is None, "ERROR; command failed."
    output = output_b.decode('utf-8')
    return output

def testOriginalModel(netPath,modelType):
    setTestCommand = testCommandTemplate.format(modelType,modelType,alSet,cfg.UUID,nIters)
    return testModel(setTestCommand)

def trainOriginalModel(nIters,modelType):
    setTrainCommand = trainOriginalCommandTemplate.format(nIters,modelType)
    output = trainModel(setTrainCommand)
    return findCaffemodelSnapshot(output)

def trainModel(setTrainCommand):
    print(setTrainCommand)
    return runCommandProcess(setTrainCommand)

def findCaffemodelSnapshot(output):
    findName = r"Wrote snapshot to: (?P<name>.*)"
    trainedModelName = re.findall(findName,output)[-1]
    return trainedModelName

def testModel(setTestCommand):
    print(setTestCommand)
    output = runCommandProcess(setTestCommand)
    findAcc = r"Accuracy: (?P<acc>[0-9]+\.[0-9]+)"
    acc = float(re.findall(findAcc,output)[0])
    return acc

def trainAlModel(alSet,nIters,ogIters,alSetDir,modelType,ogModelName):
    useSolverState = cfg.SOLVER_STATE
    infix = "{}_{}".format(alSet,cfg.UUID)
    ogIters = 2000
    if useSolverState:
        handleTrainedModelFromModelName("{}_iter_{}".format(ogModelName,ogIters))
        setTrainCommand = trainCommandTemplate_yesSolverState.format(nIters,alSet,modelType,\
                                                                     ogModelName,ogIters,\
                                                                     alSetDir,infix,
                                                                     'noCombo')
    else:
        setTrainCommand = trainCommandTemplate_noSolverState.format(nIters,alSet,modelType,\
                                                                    alSetDir,infix,
                                                                    'yesCombo')
    output = trainModel(setTrainCommand)
    print("trainAlModel")
    print(output)
    return None

def testAlModel(alSet,nIters,modelType):
    infix = "{}_{}".format(alSet,cfg.UUID)
    setTestCommand = testCommandTemplate.format(modelType,modelType,\
                                                infix,nIters)
    return testModel(setTestCommand)

def getAlSubsetInfo(alSet):
    alSetFn = alSet + ".txt"
    alSetPath = osp.join(alSetDir,alSetFn)
    alSubset = getIdList(alSetPath)
    return alSubset

def saveStatePickle(saveALResultsFn,alIds,originalAcc,subsetIndex,cover):
    print("saving state information")
    stateInfo = {"subsetIndex":subsetIndex,"cover":cover,"uuid":cfg.UUID}
    saveALResultsFnPkl = saveALResultsFn+".pkl"
    fullPath = osp.join(outputDir,saveALResultsFnPkl)
    print("saving AL state information to {}".format(saveALResultsFnPkl))
    with open(fullPath, 'wb') as f:
        pickle.dump({'alIds':alIds,'originalAcc':originalAcc,'stateInfo':stateInfo}, f)
    print("SAVED STATE INFORMATION")

def restoreStatePickle(saveALResultsFn):
    saveALResultsFnPkl = saveALResultsFn+".pkl"
    fullPath = osp.join(outputDir,saveALResultsFnPkl)
    with open(fullPath, 'rb') as f:
        stateInfo = pickle.load(f)
    return stateInfo['alIds'],stateInfo['originalAcc'],stateInfo['stateInfo']

def savePickleFile(saveALResultsFn,alIds,originalAcc):
    saveALResultsFnPkl = saveALResultsFn+".pkl"
    fullPath = osp.join(outputDir,saveALResultsFnPkl)
    print("saving raw AL results to {}".format(saveALResultsFnPkl))
    with open(saveALResultsFnPkl, 'wb') as f:
        pickle.dump({'alIds':alIds,'originalAcc':originalAcc}, f)
    

def saveCsvFileIters(saveALResultsFn,alIds,originalAcc):
    pass

def saveCsvFile(saveALResultsFn,alIds,originalAcc):
    saveALResultsFnCsv = saveALResultsFn+".csv"
    fullPath = osp.join(outputDir,saveALResultsFnCsv)
    print("saving summary stat AL results to {}".format(fullPath))
    f = open(fullPath,"w+")
    f.write("image_index,mean_acc,std_acc\n")
    for image_index,acc_list in alIds.items():
        mean = ""
        std = ""
        if acc_list is None:
            pass
            #print("warning: we actually have a none; very unlikely to occur")
        else:
            percent_diff = computePercentDifference(acc_list,originalAcc)
            print("% difference")
            print(percent_diff)
            print(acc_list)
            print(originalAcc)
            mean = np.mean(percent_diff)
            std = np.std(percent_diff)
        f.write("{},{},{}\n".format(image_index,mean,std))
    f.close()

def computePercentDifference(accuracy_list,originalAccuracy):
    acc = np.array(accuracy_list)
    oa = originalAccuracy
    return (acc - oa) / ((acc + oa) / 2.)

def handleTrainedModel(caffemodelPath):
    return
    """
    the caffemodel needs to be copied to the solverstate dir &
    the solverstate needs to be copied to the local dir
    """
    cmDir = '/'.join(caffemodelPath.split("/")[:-1])
    trainModelNetNameBase = caffemodelPath.split("/")[-1].split(".")[0]
    ss = trainModelNetNameBase+".solverstate"
    cm = trainModelNetNameBase+".caffemodel"
    # copySolverstateDest = osp.join(cmDir,ss)
    if osp.exists(ss): return ss
    copyCaffemodelDest = osp.join(cmDir,ss)
    copyCommand = "cp {} {}".format(caffemodelPath,copyCaffemodelDest)
    print(runCommandProcess(copyCommand))
    return ss

def handleTrainedModelFromModelName(modelName):
    return
    caffemodelPath = osp.join(cfg.ORIGINAL_MODEL_ROOTDIR,modelName+".caffemodel")
    print(handleTrainedModel(caffemodelPath))

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test an Object Detection network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='required config file', default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":


    args = parse_args()
    # config file
    configFn = args.cfg_file
    print(configFn)
    cfg_from_file(configFn)
    
    # arguments from cfg
    useSolverState = cfg.SOLVER_STATE
    alInfoFn = cfg.AL_INFO_FN
    nPassesofAlSubsets = cfg.NUMBER_OF_PASSES_OF_AL_SUBSET
    batchSize = cfg.BATCH_SIZE # MUST ALIGN WITH CFG FOR TRAINING THE AL_SUBSETS
    saveMod = cfg.SAVE_MOD
    restoreState = cfg.RESTORE_STATE
    modelType = cfg.AL_MODEL_TYPE

    # information from original model
    ogModelName = cfg.ORIGINAL_MODEL_NAME
    ogModelType = cfg.ORIGINAL_MODEL_TYPE
    ogIters = cfg.ORIGINAL_MODEL_ITERS
    ogAccuracy = cfg.ORIGINAL_MODEL_ACCURACY

    #-----------------------
    # CREATE ORIGINAL MODEL
    #-----------------------

    # train original model
    if useSolverState:
        if cfg.ORIGINAL_MODEL_NAME is None:
            caffemodel = trainOriginalModel(ogIters,modelType)
            solverstate,ogModelName = handleTrainedModel(caffemodel)
        else:
            ogModelName = cfg.ORIGINAL_MODEL_NAME
            caffemodel = "./output/classification/mnist/{}.caffemodel".format(ogModelName)
            solverstate = "./{}.solverstate".format(ogModelName)

    # noPrune:     0.8544 @1.4k
    # yesPrune10:  0.9337 @2k
    # yesPrune100: 0.95250 @2k
    # yesPrune200: 0.95100 @2k
    # test original model
    if cfg.ORIGINAL_MODEL_NAME is None or cfg.ORIGINAL_MODEL_ACCURACY is None:
        originalAcc = testOriginalModel(caffemodel,modelType)
    else:
        originalAcc = cfg.ORIGINAL_MODEL_ACCURACY
    print(originalAcc)

    # learning curve parameters
    startIter = cfg.AL_SUBSET_VALIDATE_START_ITER
    endIter = cfg.AL_SUBSET_VALIDATE_END_ITER
    iterFreq = cfg.AL_SUBSET_VALIDATE_FREQ

    #-------------------
    # ACTIVE LEARNING
    #-------------------

    # set al arguments from generating the al subsets
    alInfo = getAlInfo(alInfoFn)
    alSetOrigin = alInfo['alSetOrigin']
    alSetDir = alInfo['alSetDir']
    alSubsetSize = alInfo['subsetSize']
    # nIters = alInfo['subsetSize'] // batchSize * nPassesofAlSubsets + ogIters
    if useSolverState:
        nIters = endIter + ogIters
    else:
        nIters = endIter # alInfo['subsetSize'] // batchSize * nPassesofAlSubsets + ogIters

    # get of ALL image indicies available for active learning
    alIds = dict.fromkeys(getIdList(alSetOrigin),None)

    # save active learning information
    saveALResultsFn = "resultsAL_{}_{}_{}_{}_{}_ogIters{}".format(nIters,
                                                        alInfo['valSize'],
                                                        alInfo['subsetSize'],
                                                        alInfo['numberOfCovers'],
                                                        ogModelName,ogIters)
    # load state information
    cfg.UUID = str(uuid.uuid4())
    start_cover,start_subset = 0,0
    if restoreState:
        print("Restoring State with {}".format(saveALResultsFn))
        alIds,originalAcc,stateInfo = restoreStatePickle(saveALResultsFn)
        cfg.UUID = stateInfo['uuid']
        start_cover,start_subset = stateInfo['cover'],stateInfo['subsetIndex']
        print("STATE RESTORED: Starting (Cover,Subset) = ({},{})".format(start_cover,start_subset))
    
    # run the AL experiments
    for cover in range(start_cover,alInfo['numberOfCovers']):
        for subsetIndex in range(start_subset,alInfo['subsetsInCover']):
            alSet = "alSet_{}_{}_{}_{}".format(subsetIndex,cover,
                                               alInfo['subsetSize'],
                                               alInfo['valSize']) 
            trainAlModel(alSet,nIters,ogIters,alSetDir,modelType,ogModelName) # train the new model
            # get samples from learning curve
            for testIter in range(startIter,endIter+1,iterFreq):
                currIter = testIter
                if cfg.SOLVER_STATE:
                    currIter += cfg.ORIGINAL_MODEL_ITERS
                acc = testAlModel(alSet,currIter,modelType) # test the new model
                print("testing acc: {} @ {}".format(acc,currIter))
                assignAccuracy(alIds,alSet,acc,currIter,alSubsetSize) # aggregate the accuracy
            # save the current state information
            if subsetIndex % saveMod == 0:
                saveStatePickle(saveALResultsFn,alIds,originalAcc,subsetIndex,cover)
                
    # save information
    savePickleFile(saveALResultsFn,alIds,originalAcc)
    saveCsvFile(saveALResultsFn,alIds,originalAcc)


