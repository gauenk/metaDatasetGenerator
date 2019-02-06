import sys,os,pickle,uuid,cv2,glob,csv
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import numpy.random as npr
from core.config import cfg,iconicImagesFileFormat
from utils.base import scaleImage

def cropImageToAnnoRegion(im_orig,box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    return scaleCroppedImage(im_orig[y1:y2, x1:x2])

def scaleCroppedImage(im_orig):
    return scaleImage(im_orig,cfg.CROPPED_IMAGE_SIZE)

def scaleRawImage(im_orig):
    return scaleImage(im_orig,cfg.RAW_IMAGE_SIZE)

def addImgBorder(img,border=255):
    img[0,:,:] = border
    img[-1,:,:] = border
    img[:,0,:] = border
    img[:,-1,:] = border

def getImageWithBorder(_img,border=255,rotation=None):
    img = _img.copy()
    if cfg._DEBUG.utils.misc: print("[save_image_with_border] rotation",rotation)
    if rotation:
        angle,cols,rows = rotation[0],rotation[1],rotation[2]
        rotationMat,scale = getRotationInfo(angle,cols,rows)
        if cfg._DEBUG.utils.misc: print("[utils/misc.py] rotationMat",rotationMat)
        img_blank = np.zeros(img.shape,dtype=np.uint8)
        addImgBorder(img_blank,border=border)
        if cfg._DEBUG.utils.misc: print(img_blank.shape)
        img_blank = cv2.warpAffine(img_blank,rotationMat,(cols,rows),scale)
        img += img_blank
    addImgBorder(img,border=border)
    return img

def save_image_with_border(fn,_img,border=255,rotation=None):
    img = getImageWithBorder(_img,border=border,rotation=rotation)
    fp = osp.join(cfg.ROTATE_PATH,fn)
    cv2.imwrite(fp,img)

def concatenate_images(image1,image2,average_size=False,axis=1):
    # make the input larger along the width
    # axis = 0 or 1
    if average_size:
        # align image size on both dims
        average_shape = np.mean([image1.shape, image2.shape],axis=0,dtype=np.int)
        scaled_image1 = scaleImage(image1,average_shape)
        scaled_image2 = scaleImage(image2,average_shape)
    else:
        # align image size on "axis" dim
        average_shape = np.mean([image1.shape, image2.shape],axis=0,dtype=np.int)
        not_axis = np.abs(axis-1)
        shape1 = [average_shape[axis],average_shape[axis]]
        shape2 = [average_shape[axis],average_shape[axis]]
        shape1[not_axis] = image1.shape[not_axis]
        shape2[not_axis] = image2.shape[not_axis]
        scaled_image1 = scaleImage(image1,shape1)
        scaled_image2 = scaleImage(image2,shape2)
    concat_image = np.concatenate((scaled_image1,scaled_image2),axis=axis)
    # print("concat_image.shape",concat_image.shape)
    return concat_image

def splitImageForSiameseNet(img,axis=1,location="middle"):
    if location == "middle":
        if axis == 1:
            half_index = img.shape[1]//2
            img1 = img[:,:half_index,:]
            img2 = img[:,half_index:,:]
            return [img1,img2]
        else:
            print("[image_utils.py splitImageForSiameseNet]: can't handle axis {}".format(axis))
            exit()
    else:
        print("[image_utils.py splitImageForSiameseNet]: unknown split location {}".format(location))
        exit()

def save_image_list_to_file(image_list,append_str_l,vis=False,size=cfg.CROPPED_IMAGE_SIZE,infix=None):
    print("[./utils/image_utils.py: save_image_list_to_file]: saving images")
    useAppendStr = append_str_l is not None and len(append_str_l) == len(image_list)
    prev_img = image_list[0]
    for idx,img in enumerate(image_list):
        # print(img.max(),img.min())
        # print(img.shape)
        if img.max() <= 1: # rescaleImageValues
            img[:,:,:] *= 255
        #img[:size,:size,:] += cfg.PIXEL_MEANS
        img = img.astype(np.uint8)
        if idx >= 1:
            print(prev_img[15:17,15:17])
            print(img[15:17,15:17])
            print(np.all(prev_img == img))
            prev_img = img
        fn = "save_image_list_image"
        if infix:
            fn += "_{}".format(infix)
        if useAppendStr:
            fn += "{}_{}.png".format(idx,append_str_l[idx])
        else:
            fn += "{}.png".format(idx)

        print(fn)
        if vis is False:
            cv2.imwrite(fn,img)
        else:
            plt.imshow(img[:,:,::-1])
            plt.show()

