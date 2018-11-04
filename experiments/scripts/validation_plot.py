#!/usr/bin/env python3
import os,sys,re,yaml,subprocess,pickle
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

outputDir = "./output/al/"
testCommandTemplate = "./tools/test_net.py --imdb {datasetName}-test-default --cfg ./experiments/cfgs/cls_{datasetName}.yml --def ./models/{datasetName}/{modelType}/test.prototxt --net ./output/classification/{datasetName}/{ogModelName}_iter_{nIters}.caffemodel"
#testCommandTemplate = "./tools/test_net.py --imdb mnist-test-default --cfg ./experiments/cfgs/cls_mnist.yml --def ./models/mnist/{}/test.prototxt --net ./output/classification_al_subset/mnist/{}_iter_{}.caffemodel"

    # fn = "/home/gauenk/Documents/experiments/metaDatasetGenerator/output/classification/mnist/mnist_al_net_lenet5_iter_9900.caffemodel"

    
def runCommandProcess(setCommand):
    modelProc = subprocess.Popen(setCommand.split(' '),stdout=subprocess.PIPE)
    output_b,isSuccess = modelProc.communicate()
    assert isSuccess is None, print("ERROR; command failed.")
    output = output_b.decode('utf-8')
    return output

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

def testIterModel(nIters,modelType,ogModelName,datasetName):
    setTestCommand = testCommandTemplate.format(datasetName=datasetName,modelType=modelType,\
                                                ogModelName=ogModelName,nIters=nIters)
    return testModel(setTestCommand)

def getAlSubsetInfo(alSet):
    alSetFn = alSet + ".txt"
    alSetPath = osp.join(alSetDir,alSetFn)
    alSubset = getIdList(alSetPath)
    return alSubset

def saveStatePickle(saveALResultsFn,alIds,originalAcc,subsetIndex,cover):
    stateInfo = {"subsetIndex":subsetIndex,"cover":cover}
    saveALResultsFnPkl = saveALResultsFn+".pkl"
    fullPath = osp.join(outputDir,saveALResultsFnPkl)
    print("saving AL state information to {}".format(fullPath))
    with open(fullPath, 'wb') as f:
        pickle.dump({'alIds':alIds,'originalAcc':originalAcc,'stateInfo':stateInfo}, f)

def restoreStatePickle(saveALResultsFn):
    saveALResultsFnPkl = saveALResultsFn+".pkl"
    fullPath = osp.join(outputDir,saveALResultsFnPkl)
    with open(fullPath, 'rb') as f:
        stateInfo = pickle.load(f)
    return stateInfo['alIds'],stateInfo['originalAcc'],stateInfo['stateInfo']

def savePickleFile(saveALResultsFn,alIds,originalAcc):
    saveALResultsFnPkl = saveALResultsFn+".pkl"
    fullPath = osp.join(outputDir,saveALResultsFnPkl)
    print("saving raw AL results to {}".format(fullPath))
    with open(fullPath, 'wb') as f:
        pickle.dump({'alIds':alIds,'originalAcc':originalAcc}, f)
    
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
    """
    the caffemodel needs to be copied to the solverstate dir &
    the solverstate needs to be copied to the local dir
    """
    cmDir = '/'.join(caffemodelPath.split("/")[:-1])
    trainModelNetNameBase = caffemodelPath.split("/")[-1].split(".")[0]
    ss = trainModelNetNameBase+".solverstate"
    cm = trainModelNetNameBase+".caffemodel"
    # copySolverstateDest = osp.join(cmDir,ss)
    copyCaffemodelDest = osp.join(cmDir,ss)
    copyCommand = "cp {} {}".format(caffemodelPath,copyCaffemodelDest)
    print(runCommandProcess(copyCommand))
    return ss

if __name__ == "__main__":

    # arguments
    startIters = 2000
    endIters = 100000
    saveFreq = 2000
    datasetName = "cifar_10"
    modelType = 'lenet5'
    # ogModelName = "mnist_{}_short_train_3k_input28x28_yesImageNoise_noPrune".format(modelType)
    # ogModelName = "mnist_{}_short_train_3k_input28x28_yesImageNoise_yesPrune10".format(modelType)
    # ogModelName = "mnist_{}_short_train_3k_input28x28_yesImageNoise_yesPrune100".format(modelType)
    # ogModelName = "mnist_{}_short_train_3k_input28x28_yesImageNoise_yesPrune200".format(modelType)
    # ogModelName = "solverForTesting_3500"
    ogModelName = "cifar_10_lenet5_yesImageNoise_noPrune"
    # acquire the validation error
    accList = []
    for nIters in range(startIters,endIters+1,saveFreq):
        acc = testIterModel(nIters,modelType,ogModelName,datasetName) # test the new model
        print("testing acc: {}".format(acc))
        accList.append(acc)    

    print(accList)
    # save and plot the validation error
    saveFn = "./{datasetName}_validation_accuracy_{ogModelName}.pkl".format(datasetName=datasetName,ogModelName=ogModelName)
    fullPath = osp.join(outputDir,saveFn)
    with open(fullPath,'wb') as f:
        pickle.dump(accList,f)
    plt.plot(np.arange(len(accList)),accList,'r+')
    plt.show()
