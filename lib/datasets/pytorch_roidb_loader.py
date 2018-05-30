"""
Enables a user to utilize the roidb object in pytorch

Reference https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py

The class is a "torchvision.dataset.DatasetFolder" type
"""

# misc
import numpy as np

# metaDatasetGen
from datasets.ds_utils import compute_size_along_roidb,clean_box

# pytorch imports
import torch.utils.data as data
from torchvision import datasets
from torchvision.datasets.folder import IMG_EXTENSIONS,default_loader

class RoidbDataset(data.Dataset):

    def __init__(self, roidb, classes, transform=None, target_transform=None,
                 loader=default_loader, normalize_box=True):

        self.roidb = roidb
        self.classes = classes
        self.num_classes = len(classes)
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.roidbSizes = np.array(compute_size_along_roidb(roidb))
        print(self.roidbSizes[:30])
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self._normalize_box = normalize_box

    # re-defitinion of the index function 
    def __getitem__(self, index):
        """
        Args:
           index (int): Index

        Returns:
           tuple: (sample, target) where target is class_index of the target class.
        """
        sample,annoIndex = self.getSampleAtIndex(index)

        if sample['flipped']:
            print("FLIPPED")
            sys.exit()

        box = sample['boxes'][annoIndex]

        # (x1,y1,x2,y2) bounds correction
        if True:
            clean_box(box,sample)
            
        # load inputs and targets
        inputs,target = self.loader(sample,annoIndex)

        # transform image or target
        if self.transform is not None:
            inputs = self.transform(inputs,sample=sample,annoIndex=annoIndex)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return inputs, target
            
    def getSampleAtIndex(self,index):
        sampleIndex = np.argwhere(self.roidbSizes > index)[0][0]
        sample = self.roidb[sampleIndex]
        # which box to load
        if sampleIndex > 0:
            annoIndex = index - self.roidbSizes[sampleIndex-1]
        else:
            annoIndex = index
        return sample, annoIndex
                
    def __len__(self):
        return self.roidbSizes[-1]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def datasetID_at_index(self,index):
        sampleIndex = np.argwhere(self.roidbSizes > index)[0][0]
        return self.roidb[sampleIndex]['set']
