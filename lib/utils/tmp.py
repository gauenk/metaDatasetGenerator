import os.path as osp
import cfg

similarityInfo = {}
similarityInfo['repr_type'] = 'net'
similarityInfo['repr_fxn'] = computeNetRepresentation
#similarityInfo['repr_fxn'] = computePcaRepresentation
similarityInfo['model_def'] = None
similarityInfo['model_net'] = None
similarityInfo['netReprLayerName'] = 'cls_prob'

def prepare_roidb(roidb,similarityInfo=None):

    # build the compression neural network once (e.g. a VAE or GAN)
    if similarityInfo:
        cfg.SIMILARITY_SCORE_NET = createSimilarityNet(similarityInfo)
        
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
             for i in xrange(imdb.num_images)]
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        roidb[i]['image_id'] = imdb.image_index[i]
        roidb[i]['width'] = sizes[i][0]
        roidb[i]['height'] = sizes[i][1]
        roidb[i]['similarity']  = None
        # we hope we share repr information so it is all computed after the first roidb "similarityInfo"
        # this implies the "roidb" and the "similarityInfo['roidb']" should point to the same elements in memory
        if 'repr' not in roidb[i].keys():
            roidb[i]['repr']  = None
            roidb[i]['repr_type']  = None
        if similarityInfo:
            if 'repr' is None:
                roidb[i]['repr_type']  = similarityInfo['repr_type']
                roidb[i]['repr']  = similarityInfo['repr_fxn'](roidb[i])
            roidb[i]['similarity'] = computeImageSimilarity(roidb[i],similarityInfo)
    
def computeImageSimilarity(currentSample,similarityInfo):
    similaritySum = 0
    similarityRoidb = similarityInfo['roidb']
    for sample in similarityRoidb:
        if 'repr' not in sample.keys() or sample['repr'] is None:
            sample['repr']  = similarityInfo['repr_fxn'](sample)
        similaritySum += similarityEuclidean(currentSample['repr'],sample['repr'])
    return similaritySum /= (1. * len(similarityRoidb))

def computeNetRepresentation(image,similarityInfo):
    net = cfg.SIMILARITY_SCORE_NET
    return net.forward(img)[similarityInfo['netReprLayerName']]

def computePcaRepresentation(images,similarityInfo):
    # note this must be run on a *large* set of images to be effective... not computationally feasible for me mostly
    pass

def computeImageSimilarityVae(roidb[i],similarityRoidb):
    similaritySum = 0
    for sample in similarityRoidb:
        similaritySum = 

def createSimilarityNet(similarityInfo):
    netDefPath = similarityInfo['model_def']
    netNetPath = similarityInfo['model_net']
    return createNet(netDefPath,netNetPath)
    
def createNet(defPath,netPath,gpuId=0):
    caffe.set_mode_gpu()
    caffe.set_device(gpuId)
    net = caffe.Net(defPath, netPath, caffe.TEST)
    net.name = osp.splitext(osp.basename(netPath))[0]
    return net

def similarityEuclidean(img1,img2):
    return np.sum(img1-img2)

def similarityRaw(img1,img2,simParams):
    return similarityEuclidean(img1,img2)

def similarityPCA(img1,img2,simParams):
    pca1 = pcaImage(img1)
    pca2 = pcaImage(img2)
    return similarityEuclidean(pca1,pca2)

def similarityVAE(img1,img2,simParams):
    print("ERROR THIS FUNCTION SHOULDNT BE USED")
    vaeDefPath = simParams['vae_def']
    vaeNetPath = simParams['vae_net']
    net = createNet(vaeDefPath,vaeNetPath)
    repr1 = net.forward(img1)[simParams['vaeOutputName']]
    repr2 = net.forward(img2)[simParams['vaeOutputName']]
    return similarityEuclidean(repr1,repr2)
    
              


