
# metaDatasetGenerator imports
from core.config import cfg, cfgData, createFilenameID, createPathRepeat, createPathSetID
from datasets.imdb import imdb

# 'other' imports
import pickle
import numpy as np
import numpy.random as npr
import os.path as osp

def load_mixture_set(setID,repetition,final_size):

    allRoidb = []
    annoCounts = []
    datasetSizes = cfg.MIXED_DATASET_SIZES
    if final_size not in datasetSizes:
        raise ValueError("size {} is not in cfg.MIXED_DATASET_SIZES".format(final_size))
    sizeIndex = datasetSizes.index(final_size)
    prevSize = 0
    
    for size in datasetSizes[:sizeIndex+1]:
        # create a file for each dataset size
        pklName =createFilenameID(setID,str(repetition),str(size)) + ".pkl"
        # write pickle file of the roidb
        print(pklName)
        print(len(allRoidb))
        if osp.exists(pklName) is True:
            fid = open(pklName,"rb")
            loaded = pickle.load(fid)
            roidbs = loaded['allRoidb']
            if size == final_size: # only save the last count
                annoCounts = loaded['annoCounts']
            print_each_size(roidbs)
            allRoidb.extend(roidbs)
            fid.close()
        else:
            raise ValueError("{} does not exists".format(pklName))
        prevSize += len(loaded)
    return allRoidb,annoCounts

def print_each_size(roidb):
    sizes = [0 for _ in range(8)]
    for elem in roidb:
        sizes[elem['set']-1] += 1
    print(sizes)

def roidb_element_to_cropped_images(datum):
    cv2.load(datum['image'])

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

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

