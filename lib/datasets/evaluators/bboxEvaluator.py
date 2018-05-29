# --------------------------------------------------------
# CBAL
# Written by Kent Gauen
# --------------------------------------------------------

import os,sys
import os.path as osp
import PIL
import numpy as np
import scipy.sparse
from core.config import cfg,cfgData
from easydict import EasyDict as edict
from utils import *

class bboxEvaluator(object):
    """Image database."""

    def __init__(self, datasetName, classes, compID, salt, cacheDir,
                 imageSetPath, imageIndex, annoPath,load_annotation,
                 onlyCls=None):
        self._datasetName = datasetName
        self._classes = classes
        self._comp_id = compID
        self._salt = salt
        self._cachedir = cacheDir
        self.image_index = imageIndex
        self._annoPath = annoPath
        self._load_annotation = load_annotation
        self._imageSetPath = imageSetPath
        self._imageSet = imageSetPath.split("/")[-1].split(".")[0] # "...asdf/imageSet.txt"
        self._onlyCls = onlyCls

    def evaluate_detections(self, all_boxes, output_dir):
        if self._classes != num_outputs:
            raise ValueError("ERROR: the classes and output size don't match!")
        self._pathResults = output_dir
        self._write_results_file(all_boxes)
        self._do_python_eval(output_dir)

    def _write_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self._classes):
            count = 0
            skip_count = 0
            det_count = 0
            if cls == '__background__':
                continue
            print('Writing {} {} results file'.format(cls,self._datasetName))
            filename = self._get_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        skip_count+=1
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output'):
        annopath = self._annoPath + "/{:s}"
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not osp.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            if self._onlyCls is not None and cls != self._onlyCls:
                continue
            detfile = self._get_results_file_template().format(cls)
            rec, prec, ap, ovthresh = self.bbox_eval(
                detfile, annopath, self._imageSetPath, cls, self._cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
        aps = np.array(aps)
        infix = "faster-rcnn"
        if cfgData.MODEL:
            infix = cfgData.MODEL

        results_fd = open("./results_{}_{}.txt".format(infix,self._datasetName + self._salt),"w")
        for kdx in range(len(ovthresh)):
            #print('{0:.3f}@{1:.2f}'.format(ap[kdx],ovthresh[kdx]))
            print('Mean AP = {:.4f} @ {:.2f}'.format(np.mean(aps[:,kdx]),ovthresh[kdx]))
        print('~~~~~~~~')
        print('Results:')
        count_ = 1
        sys.stdout.write('{0:>15} (#):'.format("class AP"))
        results_fd.write('{0:>15} (#):'.format("class AP"))
        for thsh in ovthresh:
            sys.stdout.write("\t{:>5}{:.3f}".format("@",thsh))
            results_fd.write("\t{:>5}{:.3f}".format("@",thsh))
        sys.stdout.write("\n")
        results_fd.write("\n")
        for ap in aps:
            sys.stdout.write('{:>15} ({}):'.format(self._classes[count_],count_))
            results_fd.write('{:>15} ({}):'.format(self._classes[count_],count_))
            for kdx in range(len(ovthresh)):
                sys.stdout.write('\t{0:>10.5f}'.format(ap[kdx],ovthresh[kdx]))
                results_fd.write('\t{0:>10.5f}'.format(ap[kdx],ovthresh[kdx]))
            sys.stdout.write('\n')
            results_fd.write('\n')
            count_ +=1
        sys.stdout.write('{:>15}:'.format("mAP"))
        results_fd.write('{:>15}:'.format("mAP"))
        for kdx in range(len(ovthresh)):
            sys.stdout.write('\t{:10.5f}'.format(np.mean(aps[:,kdx])))
            results_fd.write('\t{:10.5f}'.format(np.mean(aps[:,kdx])))
            #print('{0:.3f}@{1:.2f}'.format(ap[kdx],ovthresh[kdx]))
            #print('mAP @ {:.2f}: {:.5f} '.format(ovthresh[kdx],np.mean(aps[:,kdx])))
        sys.stdout.write('\n')
        results_fd.write('\n')
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('--------------------------------------------------------------')
        print('')

    def _get_results_file_template(self):
        # example: VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._comp_id + self._salt + '_det_' + self._imageSet + '_{:s}.txt'
        path = osp.join(self._pathResults,filename)
        return path

    def bbox_eval(self,detpath,
                  annopath,
                  imagesetfile,
                  classname,
                  cachedir,
                  ovthresh=0.5,
                  use_07_metric=False):
        """rec, prec, ap = bbox_eval(detpath,
                                    annopath,
                                    imagesetfile,
                                    classname,
                                    [ovthresh],
                                    [use_07_metric])
        Top level function that does the BBOX evaluation.
        detpath: Path to detections
            detpath.format(classname) should produce the detection results file.
        annopath: Path to annotations
            annopath.format(imagename) should be the xml annotations file.
        imagesetfile: Text file containing the list of images, one image per line.
        classname: Category name (duh)
        cachedir: Directory for caching the annotations
        [ovthresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use BBOX07's 11 point AP computation
            (default False)
        """
        # assumes detections are in detpath.format(classname)
        # assumes annotations are in annopath.format(imagename)
        # assumes imagesetfile is a text file with each line an image name
        # cachedir caches the annotations in a pickle file


        # first load gt
        imagenames, recs = load_groundTruth(cachedir,imagesetfile,annopath,
                                            self._load_annotation,self._classes)
        # extract gt objects for this class
        class_recs, npos = extractClassGroundTruth(imagenames,recs,classname)
        # read dets from model
        image_ids, BB = loadModelDets(detpath,classname)

        nd = len(image_ids)
        ovthresh = [0.5,0.75,0.95]
        tp, fp = compute_TP_FP(ovthresh,image_ids,BB,class_recs)
        rec, prec, ap = compute_REC_PREC_AP(tp,fp,npos,ovthresh,classname,False)

        # print(fp,tp,rec,prec,ap,npos)
        return rec, prec, ap, ovthresh
