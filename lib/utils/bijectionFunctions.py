import os,sys,pickle
import os.path as osp
import numpy as np
import numpy.random as npr
from utils.base import readPickle,writePickle


class BijectionFunctions():

    def __init__(self, bijectionType, randomPixelsFile="./randomPixels.pkl", maxPixelNum=None):
        self.bijectionType = bijectionType
        self.randomPixelsFile = randomPixelsFile
        self.randomPixels = self.load_random_pixels()
        self.availableBijections = ['rtVal = self.random_pixel_shuffle({})']
        if maxPixelNum: self.set_random_pixels(maxPixelNum)

    def load_random_pixels(self):
        randomPixelsFile = self.randomPixelsFile
        if osp.exists(randomPixelsFile):
            return readPickle(randomPixelsFile)
        return None
        
    def set_random_pixels(self,numPixels):
        if osp.exists(self.randomPixelsFile):
            print("overwriting current randomPixelsFile: {}".format(self.randomPixelsFile))
        self.randomPixels = npr.permutation(numPixels)
        writePickle(self.randomPixelsFile,self.randomPixels)
        
    def random_pixel_shuffle(self,data):
        if self.randomPixels is None: self.set_random_pixels(data.size)
        if data.size < self.randomPixels.size:
            print("[util/bijectionFunction.py random_pixel_shuffle]:")
            print("data.size < randomPixels.size")
            sys.exit(1)
        remainder = np.arange(self.randomPixels.size,data.size)
        randomPixels = np.hstack((self.randomPixels,remainder))
        rdata = data.ravel()[randomPixels].reshape(data.shape)
        return rdata        
        
    def applyBijection(self,data):
        if self.bijectionType is None:
            return data
        if self.bijectionType not in self.availableBijections:
            print("Uknown bijection. Please use one from the list below.")
            print("-"*30)
            for name in self.availableBijections: print(name)
            print("-"*30)
            sys.exit()
        exec(self.bijectionType.format('data'))
        return rtVal


            
