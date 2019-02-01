#!/usr/bin/env python
from tools_utils import runCommandProcess
from easydict import EasyDict as edict
from fresh_config import create_model_path
import re

import _init_paths
from core.config import create_snapshot_prefix

def reset_model_name(modelInfo):
    modelInfo.name = create_snapshot_prefix(modelInfo)
    modelInfo.name += '_iter_' + str(modelInfo.iterations)
    print(modelInfo.name)
    modelInfo.net_dir = create_model_path(modelInfo)
    print(modelInfo.net_dir)
    modelInfo.net_path = modelInfo.net_dir + modelInfo.name + '.caffemodel'
    if modelInfo.class_filter:
        modelInfo.def_path = 'models/{}/{}/'.format(modelInfo.imdb_str.split('-')[0],modelInfo.architecture)
    else:
        modelInfo.def_path = 'models/{}/{}/'.format(modelInfo.imdb_str.split('-')[0],modelInfo.architecture)
    modelInfo.def_path += 'test_2cls.prototxt'
    # if modelInfo.dataset_augmentation:
    #     modelInfo.def_path += 'test_corg.prototxt'
    # else:
    #     modelInfo.def_path += 'test2.prototxt'

def set_modelInfo(modelInfo,arch,optim,train_imdb_str,noise,prune,da_aug,class_filter,iters):
    modelInfo.architecture = arch
    modelInfo.imdb_str = train_imdb_str
    modelInfo.image_noise = noise
    modelInfo.prune = prune
    modelInfo.optim = optim
    modelInfo.iterations = iters
    modelInfo.dataset_augmentation = da_aug
    modelInfo.class_filter = class_filter
    reset_model_name(modelInfo)
    
    # test_set_list = ['cifar_10-train-default','mnist-train-default',\
    #                  'cifar_10-test-default','mnist-test-default']
    # # --> model information <--
    # architecture_list = ['lenet5','vgg16']
    # iters_to_test = [(idx+1)*10000 for idx in range(10)]
    # rotation_list = [0]
    # # rotation_list = [0,15,30,45,60,75,90]
    # train_set_list = ['cifar_10','mnist']
    # image_noise_list = ['noImageNoise']
    # prune_list = ['noPrune']

def test_all_with_file(modelInfo,architecture_list,optim_list,train_set_list,test_set_list,image_noise_list,prune_list,da_aug_list,class_filter_list,iters_list,filename):
    if filename is None: print("not saving output to file")
    else:
        fid = open(filename,'a')
        add_test_all_header(fid)
        fid.close()
    test_all(modelInfo,architecture_list,optim_list,train_set_list,test_set_list,image_noise_list,prune_list,da_aug_list,class_filter_list,iters_list,filename=filename)

def add_test_all_header(fid):
    fid.write("acc,train_set,test_set,arch,optim,noise,prune,aug,class_filter,iters\n")
    
def add_result_to_file(filename,text,train_set,test_set,arch,optim,noise,prune,aug,class_filter,iters):
    if filename is None: return
    regex = r".*overall accuracy: (?P<acc>[0-9.]+).*"
    result = re.findall(regex,text)
    acc = str(result[0])
    fid = open(filename,'a+')
    #acc = result.groupdict()['acc']
    fid.write("{},{},{},{},{},{},{},{},{},{}\n".format(acc,train_set,test_set,arch,optim,noise,prune,aug,class_filter,iters))
    fid.close()

def test_all(modelInfo,architecture_list,optim_list,train_set_list,test_set_list,image_noise_list,prune_list,da_aug_list,class_filter_list,iters_list,filename=None):

    iters_list_noDsAug = [(idx+1)*700 for idx in range(40)]
    iters_list_yesDsAug10_0 = [(idx+1)*1200 for idx in range(37)] # [1000,2000,5000,10000,15000,20000]
    iters_list_yesDsAug25_0 = [2500,5000,10000,15000,20000]
    iter_list_dict = {False:iters_list_noDsAug,'10-0':iters_list_yesDsAug10_0,'25-0':iters_list_yesDsAug25_0}

    cmd_template = './tools/test_net.py --imdb {} --def {} --net {} --cfg ./experiments/cfgs/cls_{}_aug100.yml'
    # cmd_template = './tools/test_net.py --imdb {} --def {} --net {} --cfg ./experiments/cfgs/cls_{}.yml'
    for arch in architecture_list:
        for optim in optim_list:
            for train_imdb_str in train_set_list:
                for test_set in test_set_list:
                    for noise in image_noise_list:
                        for prune in prune_list:
                            for da_aug in da_aug_list:
                                for class_filter in class_filter_list:
                                    iters_list = iter_list_dict[da_aug]
                                    for iters in iters_list:
                                        test_set_name = test_set.split('-')[0]
                                        set_modelInfo(modelInfo,arch,optim,train_imdb_str,noise,prune,da_aug,class_filter,iters)
                                        cmd = cmd_template.format(test_set,modelInfo.def_path,modelInfo.net_path,test_set_name)
                                        print(cmd)
                                        result = runCommandProcess(cmd)
                                        add_result_to_file(filename,result,train_imdb_str,test_set,arch,optim,noise,prune,da_aug,class_filter,iters)
                            

def test_over_iterations(iters_to_test,rotation,av_nosave,filename=None):
    imgsets = ['cifar_10-train-default','cifar_10-val-default']
    # imgsets = ['cifar_10-val-default']
    path_template = './output/classification/cifar_10/cifar_10_lenet5_yesImageNoise_noPrune_iter_{}.caffemodel'
    print(rotation)
    cmd_template = './tools/test_net.py --imdb {} --def ./models/cifar_10/lenet5/test_corg.prototxt --net {} --cfg ./experiments/cfgs/cls_cifar_10.yml --rot '+str(rotation)
    if av_nosave: cmd_template+=' --av_nosave'
    for imgset in imgsets:
        for iters in iters_to_test:
            path = path_template.format(iters)
            cmd = cmd_template.format(imgset,path)
            print(cmd)
            result = runCommandProcess(cmd)
            add_result_to_file_skinny(filename,result,imgset,iters,rotation)
            print("-"*100)
            print(iters)
            print(result)
            print("*"*100)        

def test_over_rotations(rotation_list,iters_to_test):
    fid = open('rotation_results.txt','w+')
    for rotation in rotation_list:
        test_over_iterations(iters_to_test,rotation,True,filename=filename)
    fid.close()

def add_result_to_file_skinny(filename,text,imdb_str,iters,rotation):
    if filename is None: return
    fid = open(filename,'a+')
    regex = r".*Accuracy: (?P<acc>[0-9.]+).*"
    result = re.findall(regex,text)
    acc = str(result[0])
    #acc = result.groupdict()['acc']
    fid.write("{},{},{},{}".format(imdb_str,iters,rotation,acc))
    fid.close()

def load_text_file(filename):
    data = []
    with open(filename,'r+') as f:
        for line in f.readlines():
            data.append(line)
if __name__ == "__main__":

    # generic experiment
    # --> experiment information <--

    test_set_augmentation = [] #TODO: how to request full augmentation?
    
    # --> model information <--
    #architecture_list = ['lenet5','vgg16']
    # architecture_list = ['lenet5','highwayNet']
    # architecture_list = ['lenet5']
    # architecture_list = ['logit']
    architecture_list = ['logit','logit_with_affine']

    # architecture_list = ['highwayNet']
    #train_set_list = ['cifar_10-train-default','mnist-train-default']
    # train_set_list = ['mnist-train-default']
    # train_set_list = ['mnist-train-default']
    # train_set_list = ['mnist-train_1k_2cls-default']
    train_set_list = ['cifar_10-train-default']
    # image_noise_list = [False,2,10]
    image_noise_list = [False]
    # prune_list = [False,10,200]
    prune_list = [False]

    # iters_list = [(idx+1)*100 for idx in range(50)]
    # iters_list = [(idx+1)*2500 for idx in range(50)]

    optim_list = ['adam']
    # da_aug_list = [False,'10-0','25-0']
    # da_aug_list = ['25-0']
    #da_aug_list = [False,'10-0','25-0']
    #da_aug_list = ['10-0']
    da_aug_list = [False]
    # class_filter_list = [2]
    class_filter_list = [2]

    # test_set_list = ['cifar_10-train-default','mnist-train-default',\
    #                  'cifar_10-test-default','mnist-test-default']
    # test_set_list = ['mnist-train-default','mnist-test-default']
    # test_set_list = ['mnist-val-default']
    # test_set_list = ['mnist-val-default']
    test_set_list = ['cifar_10-val-default']

    iters_list = []

    modelInfo = edict()
    modelInfo.imdb_str = train_set_list[0].replace('-','_')
    modelInfo.architecture = architecture_list[0]
    modelInfo.optim = optim_list[0]
    modelInfo.prune = prune_list[0]
    modelInfo.image_noise = image_noise_list[0]
    modelInfo.dataset_augmentation = da_aug_list[0]
    modelInfo.class_filter = class_filter_list[0]
    modelInfo.iterations = None

    filename = 'test_all.txt'
    test_all_with_file(modelInfo,architecture_list,optim_list,train_set_list,test_set_list,
                       image_noise_list,prune_list,da_aug_list,class_filter_list,iters_list,
                       filename=filename)


    # iters_to_test = [(idx+1)*10000 for idx in range(10)]
    # test_over_iterations(iters_to_test,0,False)

    # rotation experiment

    # iters_to_test = [(idx+1)*10000 for idx in range(10)]
    # rotation_list = [0,15,30,45,60,75,90]
    # test_over_rotations(rotation_list,iters_to_test)
