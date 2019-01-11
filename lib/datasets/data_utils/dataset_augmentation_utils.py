import cv2
import numpy as np

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

def applyDatasetAugmentation(input_img,config):
    transforms = config['transformations']
    rotateInfo,translateInfo,cropInfo,flipInfo = transforms
    img = input_img.copy()
    img,_ = flipImage(img,flipInfo['flip'])
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
