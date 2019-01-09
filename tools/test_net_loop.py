#!/usr/bin/env python
from easydict import EasyDict as edict
from fresh_config import create_model_name,create_model_path
import subprocess,re

import _init_paths
from core.config import create_snapshot_prefix


def reset_model_name(modelInfo):
    modelInfo.name = create_snapshot_prefix(modelInfo)
    modelInfo.name += '_iter_' + str(modelInfo.iterations)
    print(modelInfo.name)
    modelInfo.net_dir = create_model_path(modelInfo)
    print(modelInfo.net_dir)
    modelInfo.net_path = modelInfo.net_dir + modelInfo.name + '.caffemodel'
    modelInfo.def_path = 'models/{}/{}/test_corg.prototxt'.format(modelInfo.train_set,
                                                             modelInfo.architecture)

def set_modelInfo(modelInfo,arch,optim,train_set,noise,prune,da_aug,class_filter,iters):
    modelInfo.architecture = arch
    train_set_name,train_set_split,train_set_config = train_set.split('-')
    modelInfo.train_set = train_set_name
    modelInfo.image_noise = noise
    modelInfo.prune = prune
    modelInfo.optim = optim
    modelInfo.iterations = iters
    modelInfo.dataset_augmentation = da_aug
    modelInfo.classFilter = class_filter
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

def runCommandProcess(setCommand):
    modelProc = subprocess.Popen(setCommand.split(' '),stdout=subprocess.PIPE)
    output_b,isSuccess = modelProc.communicate()
    assert isSuccess is None, "ERROR; command failed."
    output = output_b.decode('utf-8')
    return output

def test_all_with_file(modelInfo,architecture_list,optim_list,train_set_list,test_set_list,
                       image_noise_list,prune_list,da_aug_list,class_filter_list,iters_list,
                       filename):
    if filename is None: print("not saving output to file")
    else:
        fid = open('filename','a+')
        add_test_all_header(fid)
        fid.close()
    test_all(modelInfo,architecture_list,optim_list,train_set_list,test_set_list,image_noise_list,prune_list,da_aug_list,class_filter_list,iters_list,filename=filename)

def add_test_all_header(fid):
    fid.write("acc,train_set,test_set,arch,optim,noise,prune,aug,class_filter,iters\n")
    
def add_result_to_file(filename,text,train_set,test_set,arch,optim,noise,prune,aug,class_filter,iters):
    if filename is None: return
    fid = open(filename,'a+')
    regex = r".*Accuracy: (?P<acc>[0-9.]+).*"
    result = re.findall(regex,text)
    acc = str(result[0])
    #acc = result.groupdict()['acc']
    fid.write("{},{},{},{},{},{},{},{},{},{}\n".format(acc,train_set,test_set,arch,optim,noise,prune,aug,class_filter,iters))
    fid.close()

def test_all(modelInfo,architecture_list,optim_list,train_set_list,test_set_list,image_noise_list,prune_list,da_aug_list,class_filter_list,iters_list,filename=None):
    cmd_template = './tools/test_net.py --imdb {} --def {} --net {} --cfg ./experiments/cfgs/cls_{}_aug100.yml --av_nosave'
    for arch in architecture_list:
        for optim in optim_list:
            for train_set in train_set_list:
                for test_set in test_set_list:
                    for noise in image_noise_list:
                        for prune in prune_list:
                            for da_aug in da_aug_list:
                                for class_filter in class_filter_list:
                                    for iters in iters_list:
                                        test_set_name = test_set.split('-')[0]
                                        set_modelInfo(modelInfo,arch,optim,train_set,noise,prune,da_aug,class_filter,iters)
                                        cmd = cmd_template.format(test_set,modelInfo.def_path,modelInfo.net_path,test_set_name)
                                        print(cmd)
                                        result = runCommandProcess(cmd)
                                        add_result_to_file(filename,result,train_set,test_set,arch,optim,noise,prune,da_aug,class_filter,iters)
                            
    # architecture_list = ['lenet5','vgg16']
    # iters_to_test = [(idx+1)*10000 for idx in range(10)]
    # train_set_list = ['cifar_10','mnist']
    # image_noise_list = ['noImageNoise']
    # prune_list = ['noPrune']


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

# def read_and_plot(filename):
#     fid = open(filename,"r")
    

if __name__ == "__main__":

    # TODO : verify augmentations (N_SAMPLES) works as expected.

    # generic experiment
    # --> experiment information <--
    # test_set_list = ['cifar_10-train-default','mnist-train-default',\
    #                  'cifar_10-test-default','mnist-test-default']
    #test_set_list = ['mnist-train-default','mnist-test-default']
    test_set_list = ['mnist-test-default']
    test_set_augmentation = [] #TODO: how to request full augmentation?
    
    # --> model information <--
    #architecture_list = ['lenet5','vgg16']
    # architecture_list = ['lenet5','highwayNet']
    architecture_list = ['lenet5']
    #train_set_list = ['cifar_10-train-default','mnist-train-default']
    train_set_list = ['mnist-train-default']
    # image_noise_list = [False,2,10]
    image_noise_list = [False]
    # prune_list = [False,10,200]
    prune_list = [False]
    # iters_list = [(idx+7)*7500 for idx in range(4)]
    # iters_list += [(idx+30)*7500 for idx in range(10)]
    iters_list = [(idx+1)*2*7500 for idx in range(25)]
    # iters_list = [(idx+1)*10000+50000 for idx in range(50)]
    # iters_list =  [42000,84000,120000,210000,420000,840000,1002000,1410000]
    optim_list = ['adam']
    # da_aug_list = ['1-0','10-0','25-0']
    da_aug_list = [False]
    class_filter_list = [2]

    modelInfo = edict()
    modelInfo.architecture = "lenet5"
    modelInfo.iterations = 40000 #100000
    modelInfo.train_set = 'cifar_10'
    modelInfo.image_noise = 'yesImageNoise'
    modelInfo.prune = 'noPrune'
    modelInfo.optim = optim_list[0]
    
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
