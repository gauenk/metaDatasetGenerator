from fast_rcnn.nms_wrapper import nms
import numpy as np
import cPickle
import os
from cache.test_results_cache import TestResultsCache


class aggregateModelOutput():
    
    def __init__(self,imdb,num_samples,output_dir,task,score_thresh,nms_thresh,max_dets_per_image,visualize_bool,cfg):
        self.visualize_bool = visualize_bool
        self.score_thresh = score_thresh
        self.num_samples = num_samples
        self.nms_thresh = nms_thresh
        self.task = task
        self.results = None
        self.det_file = None
        self.save_key = None
        self.init_results_obj(imdb,output_dir) # sets above variables
        self.classes = imdb.classes
        self.num_classes = imdb.num_classes
        self.max_dets_per_image = max_dets_per_image
        root_dir = output_dir
        self.save_cache = TestResultsCache(output_dir,None,cfg,imdb.config)

    def init_results_obj(self,imdb,output_dir):
        if self.task == 'object_detection':
            all_boxes = [[[] for _ in xrange(self.num_samples)]
                         for _ in xrange(imdb.num_classes)]
            self.results = all_boxes
            self.det_file = os.path.join(output_dir, 'probs.pkl')
            self.save_key = 'all_boxes'
        elif self.task == 'classification':
            all_probs = [[-1 for _ in xrange(self.num_samples)]
                         for _ in xrange(imdb.num_classes)]
            self.results = all_probs
            self.det_file = os.path.join(output_dir, 'probs.pkl')
            self.save_key = 'all_probs'
        else:
            raise ValueError("unknown task [agg_model_output.py]: {}".format(self.task))

    def aggregate(self,model_output,sample_index):
        if self.task == 'object_detection':
            self.aggregateDetections(model_output,sample_index)
        elif self.task == 'classification':
            self.aggregateClassification(model_output['scores'],sample_index)
        else:
            raise ValueError("unknown task [agg_model_output.py]: {}".format(self.task))

    def load(self):
        return self.save_cache.load()

    def save(self,transformation_list):
        print(len(self.results))
        self.save_cache.save(self.results)
        # save_dict = {}
        # self.save_cache
        # save_dict[self.save_key] = self.results
        # save_dict['transformations'] = transformation_list
        # with open(self.det_file, 'wb') as f:
        #     cPickle.dump(save_dict, f, cPickle.HIGHEST_PROTOCOL)

    def aggregateClassification(self,scores,sample_index):
        # handle special case
        if scores.size == 1:
            self.results[0][sample_index] = float(scores)
            return
        scores = np.squeeze(scores)
        if self.num_classes == 1:
            self.results[0][sample_index] = float(scores[0])
        else:
            for class_index in xrange(0, self.num_classes):
                self.results[class_index][sample_index] = float(scores[class_index])

    def aggregateDetections(self,model_output,sample_index):
        scores = model_output['scores']
        boxes = model_output['scores']
        for class_index in xrange(1, self.num_classes):
            inds = np.where(scores[:, j] > self.score_thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, self.nms_thresh)
            cls_dets = cls_dets[keep, :]
            if vis:
                vis_detections(im, self.classes[class_index], cls_dets)
            self.results[class_index][sample_index] = cls_dets
        # Limit to max_per_image detections *over all classes*
        if self.max_dets_per_image:
            image_scores = np.hstack([self.results[class_index][sample_index][:, -1] for class_index in xrange(1, self.num_classes)])
            if len(image_scores) > self.max_dets_per_image:
                image_thresh = np.sort(image_scores)[-self.max_dets_per_image]
                for class_index in xrange(1, self.num_classes):
                    keep = np.where(self.results[class_index][sample_index][:, -1] >= image_thresh)[0]
                    self.results[class_index][sample_index] = self.results[class_index][sample_index][keep, :]

    def visualizeCheck(self,vis_override):
        if vis_override is True:
            return True
        else:
            return self.visualize_bool

    def visualizeDetectionsByClass(self,scores,boxes,image,vis_override=False):
        if self.visualizeCheck(vis_override) is False: return
        import matplotlib
        matplotlib.use('Agg')
        for class_index in xrange(1, self.num_classes):
            inds = np.where(scores[:, class_index] > self.score_thresh)[0]
            cls_scores = scores[inds, class_index]
            cls_boxes = boxes[inds, class_index*4:(j+1)*class_index]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = nms(cls_dets, self.nms_thresh)
            cls_dets = cls_dets[keep, :]
            cls_name = self.classes[class_index]
            visualizeDetectionsOneClass(image, cls_dets, cls_name)

    def visualizeDetectionsOneClass(self,image, dets, class_name, thresh=0.3,vis_override=False):
        """Visual debugging of detections."""
        if self.visualizeCheck(vis_override) is False: return
        import matplotlib.pyplot as plt
        im = im[:, :, (2, 1, 0)]
        for i in xrange(np.minimum(10, dets.shape[0])):
            bbox = dets[i, :4]
            score = dets[i, -1]
            if score > thresh:
                plt.cla()
                plt.imshow(im)
                plt.gca().add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor='g', linewidth=3)
                    )
                plt.title('{}  {:.3f}'.format(class_name, score))
                plt.show()
