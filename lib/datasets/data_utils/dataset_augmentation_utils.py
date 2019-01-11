import cv2
import numpy as np
from utils.base import scaleImage

def getRotationScale(M,rows,cols):
    a = np.array([cols,0,1])
    b = np.array([0,0,1])
    ta = np.matmul(M,a)
    tb = np.matmul(M,b)
    correctTranslatedIndex(ta,rows,cols)
    correctTranslatedIndex(tb,rows,cols)
    scale_a_0 = rows / ( 2. * np.abs(ta[0]) + rows )
    scale_a_1 = rows / ( 2. * np.abs(ta[1]) + rows )
    scale_b_0 = cols / ( 2. * np.abs(tb[0]) + cols )
    scale_b_1 = cols / ( 2. * np.abs(tb[1]) + cols )
    scale_list = [scale_a_0,scale_a_1,scale_b_0,scale_b_1]
    scale = np.min([scale_a_0,scale_a_1,scale_b_0,scale_b_1])
    return scale

def getRotationInfo(angle,cols,rows):
    rotationMat = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1.0)
    scale = getRotationScale(rotationMat,rows,cols)
    rotationMat = cv2.getRotationMatrix2D((cols/2,rows/2),angle,scale)
    return rotationMat,scale

def rotateImage(img,angle):
    # print('angle',angle)
    im_shape = img.shape
    rows,cols = img.shape[:2]
    rotationMat, scale = getRotationInfo(angle,cols,rows)
    img = cv2.warpAffine(img,rotationMat,(cols,rows),scale)
    rotateInfo = [angle,cols,rows,im_shape]
    return img,rotateInfo

def translateImage(img,step,direction):
    if direction == 'u': x_step,y_step=0,step
    elif direction == 'd': x_step,y_step=0,-step
    elif direction == 'l': x_step,y_step=-step,0
    elif direction == 'r': x_step,y_step=step,0
    else: raise ValueError("[translateImage]: direction not found")
    im_shape = img.shape
    rows,cols = img.shape[:2]
    scale = 1.0
    translateMat = np.array([[1,0,x_step],[0,1,y_step]],dtype=np.float)
    # print(translateMat)
    timg = cv2.warpAffine(img,translateMat,(cols,rows),scale)
    translateInfo = [step,cols,rows,im_shape]
    return timg,translateInfo

def cropImage(img,step):
    img_shape = img.shape
    if step == 0: return img
    # print("[cropImage]",step)
    timg = img[step:-step,step:-step,:]
    timg = scaleImage(timg,img_shape[0])
    return timg

def flipImage(img,flip_bool):
    if flip_bool:
        img = img[:,-1,:]
    return img

def applyDatasetAugmentation(input_img,config):
    transforms = config['transformations']
    flipInfo,translateInfo,rotateInfo,cropInfo = transforms
    img = input_img.copy()
    img = flipImage(img,flipInfo['flip'])
    img,_ = translateImage(img,translateInfo['step'],translateInfo['direction'])
    img,_ = rotateImage(img,rotateInfo['angle'])
    img = cropImage(img,cropInfo['step'])
    return img

def applyDatasetAugmentationList(input_img,configs):
    # we need a 'mesh' of transformations to use;
    # this should be (sent in as):
    # (i) fix it from random for all images in the dataset
    # (ii) fix it from random for all images in a batch
    # (iii) fix it from random for each image 
    # (iv) exhaustive list of all transformations for each image
    img_index = 0
    transform_img_list = [ None for _ in range(len(configs['transformations'][0])) ]
    # order of the triple in the list is important
    for translateInfo,rotateInfo,cropInfo in zip(*configs['transformations']):
        img = input_img.copy()
        img,_ = translateImage(img,translateInfo['step'],translateInfo['direction'])
        img,_ = rotateImage(img,rotateInfo['angle'])
        img = cropImage(img,cropInfo['step'])
        transform_img_list[img_index] = img.copy()
        img_index += 1
    return transform_img_list


#
# helper functions for more important functions
#


# start: rotation helpers

def overflowOnly(coordinate,rows,cols):
    if 0 > coordinate[0]: coordinate[0] = np.abs(coordinate[0])
    elif rows < coordinate[0]: coordinate[0] = rows - coordinate[0]
    if 0 > coordinate[1]: coordinate[1] = np.abs(coordinate[1])
    elif cols < coordinate[1]: coordinate[1] = cols - coordinate[1]

def zeroInTheRegion(coordinate,rows,cols):
    if 0 <= coordinate[0] and coordinate[0] <= rows: coordinate[0] = 0
    if 0 <= coordinate[1] and coordinate[1] <= cols: coordinate[1] = 0

def correctTranslatedIndex(coordinate,rows,cols):
    zeroInTheRegion(coordinate,rows,cols)
    overflowOnly(coordinate,rows,cols)

# end:
