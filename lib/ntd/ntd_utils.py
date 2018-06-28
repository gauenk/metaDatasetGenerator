
# metaDatasetGenerator imports
from core.config import cfg, cfgData, createFilenameID, createPathRepeat, createPathSetID
from datasets.imdb import imdb

# 'other' imports
import pickle
import numpy as np
import numpy.random as npr
import os.path as osp

import matplotlib
matplotlib.use("Agg")

from core.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, loadDatasetIndexDict,iconicImagesFileFormat
from datasets.factory import get_repo_imdb
from datasets.ds_utils import load_mixture_set,print_each_size,computeTotalAnnosFromAnnoCount,cropImageToAnnoRegion,roidbSampleHOG,roidbSampleImage,roidbSampleImageHOG
import os.path as osp
import datasets.imdb
import argparse
import pprint
import numpy as np
import matplotlib.pyplot as plt
import sys,os,cv2,pickle,uuid
# pytorch imports
from datasets.pytorch_roidb_loader import RoidbDataset
from numpy import transpose as npt
from ntd.hog_svm import plot_confusion_matrix, extract_pyroidb_features,appendHOGtoRoidb,split_data, scale_data,train_SVM,findMaxRegions, make_confusion_matrix,appendHOGtoRoidbDict,split_tr_te_data
from utils.misc import *

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def roidbToFeatures(roidb,pyloader=roidbSampleHOG,calcHog=False,roidbSizes=None):
    pyroidb = RoidbDataset(roidb,[0,1,2,3,4,5,6,7],
                           loader=pyloader,
                           transform=None)
    if roidbSizes is not None:
        pyroidb.roidbSizes = np.arange(len(roidb)) + 1
    l_feat,l_idx,y = extract_pyroidb_features(pyroidb, 'hog', cfg.clsToSet, calc_feat = calcHog, \
                                              spatial_size=(32, 32),hist_bins=32, \
                                              orient=9, pix_per_cell=8, cell_per_block=2, \
                                              hog_channel=0)
    return l_feat,l_idx,y

def mangleTestingData(l_feat_te,l_idx_te,y_te,X_test,y_test,X_idx):
    
    """
    Goal: to replace the indicies with setIDs associated with the datasets in the
    "test" section of the mixed dataset from the "train" to the "test" features

    testIndex: the index from the 
    yIndicies: a python dictionary; {"setID": list of indicies associated with the set; the indicies are the location of a sample from the set in the original testing set X_test}
       -> an element in the list gives index of the next "setID" in the current testing data
       ->
    l_feat_te: a list of hog features. 
       -> axis=0 is datasets
       -> axis=1 is hog features for a specific dataset
       -> lengths across axis=1 varies
    y_te: a list of setIDs from the "testing" section of the mixed dataset
    l_idx_te: locations of the sample in the original roidb
       -> axis=0 is datasets
       -> axis=1 is the sample location 

    idx: what use the "idx" from across the y_te?

    **error case**: if the # of training examples loaded in y_test > available # of testing
       -> shouldn't happend since the test/train split comes originally from a training set (at least) x2 the testing size
    """

    print(len(y_te))
    print(len(l_idx_te))
    print(len(l_feat_te))
    for i in range(8):
        print(len(l_idx_te[i]))
        print(len(l_feat_te[i]))

    # replace the X_test for each match of y_test
    yIndicies = {}
    dsIndicies = [ 0 for _ in range(len(l_idx_te)) ]
    for setID in y_te:
        if setID not in yIndicies.keys():
            yIndicies[setID] = list(np.where(y_test == setID)[0]) # find where the setID's are
            print("{}: {}".format(setID,len(yIndicies[setID])))
        if len(yIndicies[setID]) == 0: continue
        dsIdx = dsIndicies[setID] # index for l_feat_te
        testIndex = yIndicies[setID][0] # index for x_test
        X_test[testIndex] = l_feat_te[setID][dsIdx] # replace sample content
        X_idx[testIndex] = {"idx":int(l_idx_te[setID][dsIdx]),"split":"test"} # replace the lookup
        dsIndicies[setID] += 1 # incriment
        yIndicies[setID].remove(testIndex) # "incriment" index by removing element
    print(dsIndicies)
    
def roidbToSVMData(roidbTr,roidbTe,train_size,test_size,loaderSettings):
    ds_feat_tr,l_idx_tr,y_tr = roidbToFeatures(roidbTr,pyloader=loaderSettings['pyloader'],
                                              calcHog=loaderSettings['calcHog'],
                                              roidbSizes=loaderSettings['roidbSizes'])
    
    """
    X_train, X_test, y_train, y_test, X_idx = split_data(train_size, test_size, \
                                                         l_feat_tr,l_idx_tr, y_tr,\
                                                         loaderSettings['dsHasTest'])
    """
    ds_feat_te,l_idx_te,y_te = roidbToFeatures(roidbTe,pyloader=loaderSettings['pyloader'],
                                              calcHog=loaderSettings['calcHog'],
                                              roidbSizes=loaderSettings["roidbSizes"])
    X_train, X_test, y_train, y_test, testing_idx = split_tr_te_data(ds_feat_tr,l_idx_tr,y_tr,
                                                                     ds_feat_te,l_idx_te,y_te,
                                                                     train_size, test_size,
                                                                     loaderSettings['dsHasTest'])

    print("-=-=- training dataset counts -=-=-")
    for idx,feat in enumerate(ds_feat_tr):
        print("{}: {}, {}".format(cfg.DATASET_NAMES_ORDERED[idx],len(feat),np.sum(y_train==idx)))
    print("-=-=- testing dataset counts -=-=-")
    for idx,feat in enumerate(ds_feat_te):
        print("{}: {}, {}".format(cfg.DATASET_NAMES_ORDERED[idx],len(feat),np.sum(y_test==idx)))

    # this is a work-around for the loading of a "testing" mixed dataset... overwrites the original split from the training data

    #mangleTestingData(l_feat_te,l_idx_te,y_te,X_test,y_test,testing_idx)
    X_train, X_test = scale_data(X_train, X_test)
    print(X_train.shape)
    print(y_train.shape)
    return X_train, X_test, y_train, y_test, testing_idx
        
def prepareMixedDataset(setID,repeat,size,addHOG=True):
    mixedData = load_mixture_set(setID,repeat,size)
    roidbTrDict,annoCountTr,roidbTrDict1k = mixedData["train"][0],mixedData["train"][1],mixedData["train"][2]
    roidbTeDict,annoCountTe,roidbTeDict1k = mixedData["test"][0],mixedData["test"][1],mixedData['test'][2]
    printRoidbDictImageNamesToTextFile(roidbTrDict,"train_{}".format(setID))
    printRoidbDictImageNamesToTextFile(roidbTeDict,"test_{}".format(setID))
    # does the dataset have a "testing" split?
    
    dsHasTest = [ (i is not None) and (j is not None) for i,j in zip(annoCountTr[size],
                                                                     annoCountTe[size]) ]
    # cropped hog image input
    if addHOG:
        appendHOGtoRoidbDict(roidbTrDict,size)
        appendHOGtoRoidbDict(roidbTeDict,size)
        appendHOGtoRoidbDict(roidbTrDict1k,1000)
        appendHOGtoRoidbDict(roidbTeDict1k,1000)


    print("annoCountTr: {}".format(annoCountTr[size]))
    print("annoCountTe: {}".format(annoCountTe[size]))
    # print_report(roidbTr,annoCountTr,roidbTe,annoCountTe,setID,repeat,size)
    annoSizes = {}
    annoSizes['train'] = annoCountTr
    annoSizes['test'] = annoCountTe

    print("-="*50)

    return roidbTrDict,roidbTeDict,roidbTrDict1k,roidbTeDict1k,dsHasTest,annoSizes

def loadSvmModel(modelParams,dataType,setID,repeat,size,X_train,y_train):
    modelFn = modelParams['modelFn']
    if modelFn is not None:
        model = pickle.load(open(modelFn,"rb"))
    else:
        model = train_SVM(X_train,y_train)
        fn = iconicImagesFileFormat().format("model{}_svm_{}_{}_{}.pkl".format(dataType,setID,repeat,size))
        pickle.dump(model,open(fn,"wb"))
        print(" saved model to {}".format(fn))
        
    print("\n\n-=- model loaded -=-\n\n")
    return model

def loadDlModel(modelParams,dataType,setID,repeat,size,X_train,y_train):
    pass

def genConfCropped(modelParams,roidbTr,roidbTe,ntdGameInfo):
    loaderSettings = {}
    loaderSettings['pyloader'] = roidbSampleHOG
    loaderSettings['calcHog'] = False
    loaderSettings['roidbSizes'] = None
    loaderSettings['dsHasTest'] = ntdGameInfo['dsHasTest'] # todo: kind of gross here
    return genConf(modelParams,"Cropped",roidbTr,roidbTe,loaderSettings,ntdGameInfo)

def genConfRaw(modelParams,roidbTr,roidbTe,ntdGameInfo):
    loaderSettings = {}
    loaderSettings['pyloader'] = roidbSampleImageHOG
    loaderSettings['calcHog'] = False
    loaderSettings['roidbSizes'] = np.arange(len(roidbTr)) + 1
    loaderSettings['dsHasTest'] = ntdGameInfo['dsHasTest'] # todo: kind of gross here
    return genConf(modelParams,"Raw",roidbTr,roidbTe,loaderSettings,ntdGameInfo)

def genConfSVM(modelParams,dataType,roidbTr,roidbTe,loaderSettings,ntdGameInfo):
    X_train, X_test, y_train, y_test, X_idx = roidbToSVMData(roidbTr,roidbTe,
                                                             ntdGameInfo['trainSize'],
                                                             ntdGameInfo['testSize'],
                                                             loaderSettings)
    model = loadSvmModel(modelParams,dataType,ntdGameInfo['setID'],ntdGameInfo['repeat'],
                      ntdGameInfo['size'],X_train,y_train)
    print(X_test.shape)
    print(y_test.shape)
    print("accuracy on test data {}".format(model.score(X_test,y_test)))
    print(make_confusion_matrix(model, X_train, y_train, cfg.clsToSet))
    print("-"*50)
    return make_confusion_matrix(model, X_test, y_test, cfg.clsToSet),model

def genConfDl(modelParams,dataType,roidbTr,roidbTe,loaderSettings,ntdGameInfo):
    X_train, X_test, y_train, y_test, X_idx = roidbToDlData(roidbTr,roidbTe,
                                                             ntdGameInfo['trainSize'],
                                                             ntdGameInfo['testSize'],
                                                             loaderSettings)
    model = loadDlModel(modelParams,dataType,ntdGameInfo['setID'],ntdGameInfo['repeat'],
                      ntdGameInfo['size'],X_train,y_train)
    print("accuracy on test data {}".format(model.score(X_test,y_test)))
    return make_confusion_matrix(model, X_test, y_test, cfg.clsToSet),model

def genConf(modelParams,dataType,roidbTr,roidbTe,loaderSettings,ntdGameInfo):
    modelType = modelParams['modelType']
    if modelType == "svm":
        return genConfSVM(modelParams,dataType,roidbTr,roidbTe,loaderSettings,ntdGameInfo)
    elif modelType == "dl":
        return genConfDl(modelParams,dataType,roidbTr,roidbTe,loaderSettings,ntdGameInfo)
    else:
        print("Uknown model type of {}".format(modelType))
    return None

def saveNtdSummaryStats(cmRaw_l,cmCropped_l,cmDiff_l):

    import scipy.stats as ss

    cmRaw_l = np.array(cmRaw_l)
    cmCropped_l = np.array(cmCropped_l)
    cmDiff_l = np.array(cmDiff_l)

    cmRaw_mean = np.mean(cmRaw_l,axis=0)
    cmCropped_mean = np.mean(cmCropped_l,axis=0)
    cmDiff_mean = np.mean(cmDiff_l,axis=0)

    cmRaw_std = np.std(cmRaw_l,axis=0)
    cmCropped_std = np.std(cmCropped_l,axis=0)
    cmDiff_std = np.std(cmDiff_l,axis=0)

    
    paired_tTest_num = cmRaw_mean - cmCropped_mean
    paired_tTest_denom = np.sqrt( (cmRaw_std**2 + cmCropped_std**2) / len(cmRaw_l) )
    t_values = paired_tTest_num / paired_tTest_denom
    p_values = ss.t.sf(t_values,len(cmRaw_l)-1)

    def saveMat(fn,mat):
        fid = open(iconicImagesFileFormat().format(fn),"wb")
        pickle.dump({"raw":cmRaw,"cropped":cmCropped},fid)
        fid.close()
        
    fn = "statSumm_{}_{}.pkl".format("rawMean",cfg.uuid)
    saveMat(fn,cmRaw_mean)
    fn = "statSumm_{}_{}.pkl".format("rawStd",cfg.uuid)
    saveMat(fn,cmRaw_std)
    fn = "statSumm_{}_{}.pkl".format("croppedMean",cfg.uuid)
    saveMat(fn,cmCropped_mean)
    fn = "statSumm_{}_{}.pkl".format("croppedStd",cfg.uuid)
    saveMat(fn,cmCropped_std)
    fn = "statSumm_{}_{}.pkl".format("diffMean",cfg.uuid)
    saveMat(fn,cmDiff_mean)
    fn = "statSumm_{}_{}.pkl".format("diffStd",cfg.uuid)
    saveMat(fn,cmDiff_std)
    fn = "statSumm_{}_{}.pkl".format("pValues",cfg.uuid)
    print(p_values)
    saveMat(fn,p_values)

    
    

    







