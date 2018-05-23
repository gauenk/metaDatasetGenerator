from __future__ import print_function, division

# metaDatasetGenerator imports
from core.config import cfg, cfgData, createFilenameID, createPathRepeat, createPathSetID
from datasets.imdb import imdb

# 'other' imports
import numpy as np
import numpy.random as npr
import time,os,copy
import matplotlib
matplotlib.use('Agg') # uncomment when devenloping in terminal
import matplotlib.pyplot as plt

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from utils import train_model

plt.ion()   # interactive mode

# -=-=-=-=-=-=-
# Load Data
# -=-=-=-=-=-=-

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'mainData100'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

# -=-=-=-=-=-=-=-=-=-
# Training the model
# -=-=-=-=-=-=-=-=-=-

##train from scratch############

model_ft = models.vgg16(num_classes = 8)
for param in model_ft.parameters():
    param.requires_grad = True #before true
  #  print(param.requires_grad)
   
#num_ftrs = 7
#model_ft.num_classes = 7
#model_ft.classifier[6].out_features = num_ftrs
#model_ft.classifier.modules[6] = nn.Linear(7, 7)
print(model_ft)
if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

#print(model_ft)
# Observe that all parameters are being optimized
optimizer_ft  = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
######################################################################
torch.save(model_ft.state_dict(), 'vgg16_100.pt')
