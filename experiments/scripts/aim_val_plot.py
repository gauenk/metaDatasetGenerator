#!/usr/bin/env python3
import os,sys,re,yaml,subprocess,pickle
import os.path as osp
import numpy as np
from easydict import EasyDict as edict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(dpi=800)

outputDir = "./output/aim/"
testCommandTemplate = "./tools/test_net.py --imdb cifar_10-train-default --cfg ./experiments/cfgs/classification_aim_cifar_10.yml --def ./models/tp_fn_record/aim_net/cifar_10/{}/test.prototxt --net /home/gauenk/Documents/experiments/metaDatasetGenerator/output/classification/cifar_10/{}_iter_{}.caffemodel --al_net ./output/classification/cifar_10/cifar_10_lenet5_yesImageNoise_noPrune_iter_100000.caffemodel --al_def ./models/cifar_10/lenet5/test.prototxt"

#testCommandTemplate = "./tools/test_net.py --imdb mnist-train_10k-default --cfg ./experiments/cfgs/classification_aim_mnist.yml --def ./models/tp_fn_record/aim_net/{}/test.prototxt --net /home/gauenk/Documents/experiments/metaDatasetGenerator/output/classification/mnist/{}_iter_{}.caffemodel --al_net ./output/classification/mnist/mnist_lenet5_short_train_3k_input28x28_yesImageNoise_yesPrune200_iter_4000.caffemodel --al_def ./models/mnist/lenet5/test.prototxt"

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
    tpr,tnr,ppr,npr,acc = regexMetrics(output)
    return tpr,tnr,ppr,npr,acc

def regexMetrics(outputStr):
    
    # True Positive Rate
    findMetric = r"TPR: (?P<tpr>(?:[0-9]+\.[0-9]+|nan))"
    raw = re.findall(findMetric,outputStr)[0]
    if raw == 'nan':
        raw = -1
    else:
        raw = float(raw)
    tpr = raw
    # True Negative Rate
    findMetric = r"TNR: (?P<tnr>(?:[0-9]+\.[0-9]+|nan))"
    raw = re.findall(findMetric,outputStr)[0]
    if raw == 'nan':
        raw = -1
    else:
        raw = float(raw)
    tnr = raw

    # Positive Precision Value
    findMetric = r"PPV: (?P<ppv>[0-9]+\.[0-9]+)"
    raw = re.findall(findMetric,outputStr)[0]
    if raw == 'nan':
        raw = -1
    else:
        raw = float(raw)
    ppv = raw

    # Negative Precision Value
    findMetric = r"NPV: (?P<npv>[0-9]+\.[0-9]+)"
    raw = re.findall(findMetric,outputStr)[0]
    if raw == 'nan':
        raw = -1
    else:
        raw = float(raw)
    npv = raw

    # Accuracy
    findMetric = r"ACC: (?P<acc>[0-9]+\.[0-9]+)"
    raw = re.findall(findMetric,outputStr)[0]
    if raw == 'nan':
        raw = -1
    else:
        raw = float(raw)
    acc = raw
    
    return tpr,tnr,ppv,npv,acc

def testIterModel(nIters,modelType,ogModelName):
    setTestCommand = testCommandTemplate.format(modelType,ogModelName,nIters)
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
    modelType = 'lenet5'
    ogModelName = "aim_lenet5_av400_al28"
    #ogModelName = "mnist_aim_lenet5_av400_al28"


    # acquire the validation error
    metricList = []
    for nIters in range(startIters,endIters+1,saveFreq):
        tpr,tnr,ppv,npv,acc = testIterModel(nIters,modelType,ogModelName) # test the new model
        print("testing results @ {}".format(nIters))
        print("tpr: {}".format(tpr))
        print("tnr: {}".format(tnr))
        print("ppv: {}".format(ppv))
        print("npv: {}".format(npv))
        print("acc: {}".format(acc))
        metricList.append([tpr,tnr,ppv,npv,acc])
        
    print(metricList)
    # save and plot the validation error
    saveFn = "./cifar_10_aim_validation_accuracy_{}.pkl".format(ogModelName)
    fullPath = osp.join(outputDir,saveFn)
    with open(fullPath,'wb') as f:
        pickle.dump(metricList,f)
    metricNames = ["tpr","tnr","ppv","npv","acc"]
    metricPlot = ['g^','g*','r^','r*','b-']
    metricNumpy = np.array(metricList)

    for idx,metric in enumerate(metricNames):
        mNumpy = metricNumpy[:,idx]
        print(metric)
        plt.plot(np.arange(len(mNumpy)),mNumpy,metricPlot[idx],label=metric)
        lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        saveFn = "aim_val_plot_{}.png".format(metric)
        plt.savefig(saveFn,bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.clf()
        plt.cla()

    for idx,metric in enumerate(metricNames):
        mNumpy = metricNumpy[:,idx]
        print(metric)
        plt.plot(np.arange(len(mNumpy)),mNumpy,metricPlot[idx],label=metric)
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    saveFn = "aim_val_plot.png"
    plt.savefig(saveFn,bbox_extra_artists=(lgd,), bbox_inches='tight')

        
