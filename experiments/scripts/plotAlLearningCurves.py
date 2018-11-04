#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os.path as osp

outputDir = "./output/al"
# infixStr = "mnist_lenet5_short_train_3k_input28x28_yesImageNoise_yesPrune200"
# appendStr = "resultsAL_6000_30000_500_20_{}_ogIters2000".format(infixStr)


infixStr = "mnist_lenet5_short_train_3k_input28x28_yesImageNoise_yesPrune200"
# -=-=-=-=-=-=-=-=-=-=-=-
# retrain from scratch
# -=-=-=-=-=-=-=-=-=-=-=-
# subset size: 10
# appendStr = "resultsAL_4000_30000_10_10_{}_ogIters2000".format(infixStr)
# ogMaxItersToPlot = 4000
# subset size: 500
# appendStr = "resultsAL_4000_30000_500_20_{}_ogIters2000".format(infixStr)
# ogMaxItersToPlot = 4000
# subset size: 5k
# appendStr = "resultsAL_4000_30000_5000_10_{}_ogIters2000".format(infixStr)
# ogMaxItersToPlot = 4000
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# start for current learning state
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# subset size: 10
# appendStr = "resultsAL_6000_30000_10_10_{}_ogIters2000".format(infixStr)
# ogMaxItersToPlot = 6000
# subset size: 500
appendStr = "resultsAL_6000_30000_500_20_{}_ogIters2000".format(infixStr)
ogMaxItersToPlot = 6000
# subset size: 5k
# appendStr = "resultsAL_6000_30000_5000_10_{}_ogIters2000".format(infixStr)
# ogMaxItersToPlot = 6000

def getSampleStats(sample):
    sampleMeans = []
    sampleStderrs = []
    for nIters in sorted(sample.keys()):
        nSamples = len(sample[nIters])
        mean = np.mean(sample[nIters])
        stderr = np.std(sample[nIters])/np.sqrt(nSamples)
        sampleMeans.append(mean)
        sampleStderrs.append(stderr)
    return sorted(sample.keys()),sampleMeans,sampleStderrs

def getSampleSetStats(sampleDict):
    sampleId = []
    nIter_List = []
    sampleMeans_List = []
    sampleStderrs_List = []
    for name,sample in sampleDict.items():
        if sample is None: continue
        nIter,sampleMeans,sampleStderrs = getSampleStats(sample)        
        sampleId.append(name)
        nIter_List.append(nIter)
        sampleMeans_List.append(sampleMeans)
        sampleStderrs_List.append(sampleStderrs)
    atSingleIter = None
    # atSingleIter = 3800
    if atSingleIter is not None:
        plotSampleSetStatsAtSinglePoint(sampleId, nIter_List, sampleMeans_List,\
                                        sampleStderrs_List,atSingleIter)
    else:
        plotSampleSetStats(sampleId, nIter_List, sampleMeans_List, sampleStderrs_List)        
    return sampleId, nIter_List, sampleMeans_List, sampleStderrs_List

def getOutsideDataList(pklDict):
    """
    pklDict = {"datasetID": "filename.pkl"}
    """
    dataDict = {}
    for key,pklFile in pklDict.items():
        dataDict[key] = getOriginalData(pklFile)
    return dataDict
    
def getOriginalData(fn):
    fullPath = osp.join(outputDir,fn)
    with open(fullPath,'rb') as f:
        ogSamplesRaw = pickle.load(f)
    ogSampleKeys = np.arange(200,ogMaxItersToPlot+1,200)
    #ogSamples = ogSamplesRaw[1:12:1] # when saved for every 100 iters total of 10,000
    ogSamples = ogSamplesRaw
    ogDict = {}
    ogDict['og'] = {}
    for key,value in zip(ogSampleKeys,ogSamples):
        ogDict['og'][key] = [value]
    return getSampleStats(ogDict['og'])

def plotSampleSetStats(sampleId, nIter_List, sampleMeans_List, sampleStderrs_List):

    #getSampleSetStats
    #fn = "mnist_train-default_validation_accuracy_yesImageNoise_yesPrune.pkl"

    # load original data
    if infixStr != "":
        loadFn = "mnist_validation_accuracy_{}.pkl".format(infixStr)
    else:
        loadFn = "mnist_validation_accuracy.pkl"
    ogId = "3k"
    ogX,ogY,ogYerr = getOriginalData(loadFn)
    plt.errorbar(ogX,ogY,yerr=ogYerr,label=ogId)

    
    # load & plot other validation data
    # pklDict = {"3.5k": "mnist_validation_accuracy_solverForTesting_3500.pkl"}
    # pklDict["5k"] = "mnist_validation_accuracy_solverForTesting_5000.pkl"
    # pklDict = {"test":"mnist_validation_accuracy_mnist_al_lenet5_alSet_0_0_500_30000.pkl"}
    pklDict = None
    if pklDict is not None:
        otherData = getOutsideDataList(pklDict)
        for key,val in otherData.items():
            label = key
            x = val[0]
            y = val[1]
            yerr = val[2]
            plt.errorbar(x,y,yerr=yerr,label=label)
    
    # variable to determine if we include a legend
    # countNumberPlotted = 0
    # plot all that sample data

    plotSamples = True
    if plotSamples:
        for sId, nIters, sampleMeans, sampleStderrs in zip(sampleId, nIter_List, sampleMeans_List, sampleStderrs_List):
            label = sId
            x = nIters
            y = sampleMeans
            yerr = sampleStderrs
            plt.errorbar(x,y,yerr=yerr,label=label)

    vis = False
    if appendStr != "":
        saveFn = "valExample_{}.png".format(appendStr)
    else:
        saveFn = "valExample.png"
    if vis:
        lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
    print("saving figure at: {}".format(saveFn))
    if len(sampleId) > 10 and plotSamples is True:
        plt.savefig(saveFn, bbox_inches='tight')
    else:
        lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(saveFn,bbox_extra_artists=(lgd,), bbox_inches='tight')

def plotSampleSetStatsAtSinglePoint(sampleId, nIter_List, sampleMeans_List, sampleStderrs_List,atSingleIter):

    #getSampleSetStats
    #fn = "mnist_train-default_validation_accuracy_yesImageNoise_yesPrune.pkl"

    # load original data
    if infixStr != "":
        loadFn = "mnist_validation_accuracy_{}.pkl".format(infixStr)
    else:
        loadFn = "mnist_validation_accuracy.pkl"
    ogId = 29999
    ogX,ogY,ogYerr = getOriginalData(loadFn)
    #plt.errorbar(ogX,ogY,yerr=ogYerr,label=ogId)
    iterIndex = ogX.index(atSingleIter)
    # plt.errorbar(ogId,ogY[iterIndex],yerr=ogYerr[iterIndex],label=atSingleIter)
    ogAcc = ogY[iterIndex]

    
    # load & plot other validation data
    # pklDict = {"3.5k": "mnist_validation_accuracy_solverForTesting_3500.pkl"}
    # pklDict["5k"] = "mnist_validation_accuracy_solverForTesting_5000.pkl"
    # pklDict = {"test":"mnist_validation_accuracy_mnist_al_lenet5_alSet_0_0_500_30000.pkl"}
    pklDict = None
    if pklDict is not None:
        otherData = getOutsideDataList(pklDict)
        for key,val in otherData.items():
            label = val[0]
            x = key
            y = val[1]
            yerr = val[2]
            plt.errorbar(x,y,yerr=yerr,label=label)
    
    # plot all that sample data
    plotSamples = True
    # plotSamples = True
    if plotSamples:
        for sId, nIters, sampleMeans, sampleStderrs in zip(sampleId, nIter_List, sampleMeans_List, sampleStderrs_List):
            iterIndex = nIters.index(atSingleIter)
            label = nIters[iterIndex]
            x = sId
            y = computePercentDifference([sampleMeans[iterIndex]],ogAcc)
            yerr = sampleStderrs[iterIndex]
            plt.errorbar(x,y,yerr=yerr,label=label)

    vis = False
    if appendStr != "":
        saveFn = "percentDiff_{}_atIter_{}.png".format(appendStr,atSingleIter)
    else:
        saveFn = "percentDiff_atIter_{}.png".format(atSingleIter)
    if vis:
        lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
    print("saving figure at: {}".format(saveFn))
    if len(sampleId) > 10:
        plt.savefig(saveFn, bbox_inches='tight')
    else:
        lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(saveFn,bbox_extra_artists=(lgd,), bbox_inches='tight')

def computePercentDifference(accuracy_list,originalAccuracy):
    acc = np.array(accuracy_list)
    oa = originalAccuracy
    return (acc - oa) / ((acc + oa) / 2.)

        
if __name__ == "__main__":


    #fn = "resultsAL_1200_30000_500_20.pkl"
    print("Important info")
    print("infixStr:")
    print(infixStr)
    print("appendStr:")
    print(appendStr)
    print("-="*30+"-")

    # load validation pickle file
    loadFn = "{}.pkl".format(appendStr)
    fullPath = osp.join(outputDir,loadFn)
    with open(fullPath,'rb') as f:
        tmp = pickle.load(f)

    #print(tmp)
    samples = tmp['alIds']
    # x = sorted(samples[59928].keys())
    # print(x)

    #subset of dictionary
    keys = np.arange(30000,30010)
    # keys = np.arange(40000,55000)
    # keys = np.arange(30000,60000)
    print(keys)
    subsetDict = {idx: samples[idx] for idx in keys}

    getSampleSetStats(subsetDict)
    # sampleMeans = []
    # sampleStderrs = []
    # for nIters in sorted(samples[59999].keys()):
    #     nSamples = len(samples[59999][nIters])
    #     mean = np.mean(samples[59999][nIters])
    #     stderr = np.std(samples[59999][nIters])/np.sqrt(nSamples)
    #     sampleMeans.append(mean)
    #     sampleStderrs.append(stderr)
        
    # x = sorted(samples[59999].keys())
    # y = sampleMeans
    # yerr = sampleStderrs
    # plt.errorbar(x,y,yerr=yerr)
    # plt.savefig("valExample_resultsAL_6000_30000_500_20_mnist_lenet5_short_train_3k_input28x28_yesImageNoise_yesPrune200_ogIters2000.png")

