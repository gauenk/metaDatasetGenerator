# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from core.config import cfg,prototxtToYaml,create_snapshot_prefix
import roi_data_layer.roidb as rdl_roidb
import cls_data_layer.roidb as cls_roidb
import alcls_data_layer.roidb as alcls_roidb
import aim_data_layer.roidb as aim_roidb
import vae_data_layer.roidb as vae_rdl_roidb
from utils.timer import Timer
from datasets import ds_utils
from core.prune import prune_net_iterative_step
import numpy as np
import os
import os.path as osp

from caffe.proto import caffe_pb2
import google.protobuf as pb2
import google.protobuf.text_format

from net_utils.misc import setNetworkMasksToOne
from net_utils.initialize_net import mangleSolverPrototxt,initializeNetworkWeights,setNetForWarpAffineLayerType

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, imdb, output_dir,
                 pretrained_model=None, solver_state=None,al_net=None):
        """Initialize the SolverWrapper."""
        roidb = imdb.roidb
        self.imdb = imdb

        if (cfg.TRAIN.OBJ_DET.HAS_RPN and cfg.TRAIN.OBJ_DET.BBOX_REG and
            cfg.TRAIN.OBJ_DET.BBOX_NORMALIZE_TARGETS and
            cfg.TASK == "object_detection"):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a priori
            assert cfg.TRAIN.OBJ_DET.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        if cfg.TRAIN.OBJ_DET.BBOX_REG and cfg.TASK == 'object_detection':
            print('Computing bounding-box regression targets...')
            self.bbox_means, self.bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(roidb)
            print('done')

        print(caffe.SGDSolver(solver_prototxt))
        self.solver = caffe.SGDSolver(solver_prototxt)

        if solver_state is not None:
            print("Loading solver state from {:s}".format(solver_state))
            self.solver.restore(solver_state)
        else:
            initializeNetworkWeights(self.solver.net)

        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        # handle special needs for different layer types
        setNetForWarpAffineLayerType(self.solver.net,solver_prototxt)


        # HANDLE THE MASK ISSUES (yikes)
        net = self.solver.net
        setNetworkMasksToOne(net)
        if al_net: setNetworkMasksToOne(al_net)

        # set data for dataLoader class
        if cfg.TASK == "object_detection":
            self.solver.net.layers[0].set_roidb(roidb)
        elif cfg.TASK == "classification" or cfg.TASK == "regeneration":
            if cfg.SUBTASK == "tp_fn":
                person_records = ds_utils.loadEvaluationRecords("person")
                # write_keys_to_csv(person_records)
                # write_roidb_ids_to_csv(roidb)
                # sys.exit()
                tmp = person_records.keys()
                print(len(tmp),len(roidb))
                print("they don't need to be equal")
                print("the left number has a one-to-many mapping")
                if self.solver.net.layers[0].name == "AlclsDataLayer":
                    self.solver.net.layers[0].set_roidb(roidb, person_records,al_net)
                elif self.solver.net.layers[0].name == "AimDataLayer":
                    self.solver.net.layers[0].set_roidb(roidb, person_records,al_net)
                elif self.solver.net.layers[0].name == "ClsDataLayer":
                    ds_loader = imdb.create_data_loader(cfg,person_records,al_net)
                    self.solver.net.layers[0].set_data_loader(ds_loader,imdb.data_loader_config)
                else:
                    print("WE DONT KNOW THE FIRST LAYER TYPE")
                    print(self.solver.net.layers[0].name)
                    print("PROG. QUITTING")
                    sys.exit()
            else:
                print("NOT LOADING PERSON RECORDS SINCE cfg.SUBTASK == '{}'".format(cfg.SUBTASK))
                ds_loader = imdb.create_data_loader(cfg,None,al_net)
                print("number of samples: {}".format(ds_loader.num_samples))
                print("batch size: {}".format(cfg.TRAIN.BATCH_SIZE))
                print("iterations per epoch: {}".format(ds_loader.num_samples // cfg.TRAIN.BATCH_SIZE))
                self.solver.net.layers[0].set_data_loader(ds_loader,imdb.data_loader_config)

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        scale_bbox_params = (cfg.TRAIN.OBJ_DET.BBOX_REG and
                             cfg.TRAIN.OBJ_DET.BBOX_NORMALIZE_TARGETS and
                             net.params.has_key('bbox_pred'))

        if scale_bbox_params:
            # save original values
            orig_0 = net.params['bbox_pred'][0].data.copy()
            orig_1 = net.params['bbox_pred'][1].data.copy()

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data *
                     self.bbox_stds[:, np.newaxis])
            net.params['bbox_pred'][1].data[...] = \
                    (net.params['bbox_pred'][1].data *
                     self.bbox_stds + self.bbox_means)

        # infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
        #          if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + 
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        # filename = os.path.join(self.output_dir, filename)

        # save network
        net.save(str(filename))
        print('Wrote snapshot to: {:s}'.format(filename))

        # save solverstate 
        # filename = (self.solver_param.snapshot_prefix + infix +
        #             '_iter_{:d}'.format(self.solver.iter) + '.solverstate')
        # filename = os.path.join(self.output_dir, filename)

        self.solver.save(str(filename))

        print('Wrote snapshot of solversate to: {:s}'\
            .format(filename))

        if scale_bbox_params:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1
        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        print("snapshot_prefix = [{}]".format(self.solver_param.snapshot_prefix))
        print("iteration {}/{}".format(self.solver.iter,max_iters))
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()

            if self.solver.iter % (1 * self.solver_param.display) == 0:
                if 'Sigmoid' in self.solver.net.layers[-1].type:
                    self.view_sigmoid_output()
                elif 'Softmax' in self.solver.net.layers[-1].type:
                    self.view_softmax_output()
                print('speed: {:.3f}s / iter'.format(timer.average_time))

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0 or self.solver.iter == 100:
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())
                
            if cfg.PRUNE_NET and self.solver.iter % cfg.PRUNE_NET == 0:
                prune_net_iterative_step(self.solver.net)

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths

    def print_the_last_layer(self):
        net = self.solver.net
        labels = net.blobs['labels']
        loss_cls = net.blobs['loss_cls']
        print(labels.data,loss_cls.data)

    def print_gradient(self):
        net = self.solver.net
        print(dir(net))
        net.forward()
        print(net.layer_dict.keys())
        print("-- gradients --")
        # print(net.layer_dict.keys())
        # # grads = net.layer_dict['input-data'].blobs[0].diff
        # # print(np.sum(grads),"data")
        grads = net.layer_dict['conv1_1'].blobs[0].diff
        print(np.sum(grads),"conv1_1")
        grads = net.layer_dict['_fc6'].blobs[0].diff
        print(np.sum(grads),"fc6")

        grads = net.layer_dict['_fc7'].blobs[0].diff
        print(np.sum(grads),"fc7")
        grads = net.layer_dict['cls_score'].blobs[0].diff
        print(np.sum(grads),"cls_score")
        
        # blob = net.layer_dict['_fc7'].blobs[0].data
        # print(blob,"fc7")
        # blob = net.layer_dict['cls_score'].blobs[0].data
        # print(blob,"cls_score")
        
        # grads = net.layer_dict['sm_cls'].blobs[0].diff
        # print(np.sum(grads),"sm_cls")

    def test_SoftmaxLoss(self):
        net = self.solver.net
        #print(net.blobs.keys())
        # for i in range(30):
        #     print("shape @ {}: {}".format(i,net.blobs[i].shape))
        # print(dir(net))
        # print(len(net.blobs))
        # print(dir(net.layer_dict['cls_score']))
        # sys.exit()
        #print(net.layer_dict.keys())
        lp = caffe_pb2.LayerParameter()
        lp.type = "Sigmoid"
        layer = caffe.create_layer(lp)
        data = net.blobs['cls_score'].data
        bottom = [caffe.Blob(data.shape)]
        bottom[0].data[...] = data
        top = [caffe.Blob([])]
        layer.SetUp(bottom, top)
        layer.Reshape(bottom, top)
        layer.Forward(bottom, top)
        print("data",data,"top[0].data",top[0].data)
        print(net.blobs['labels'].data - top[0].data)

    def view_sigmoid_output(self):
        net = self.solver.net
        lp = caffe_pb2.LayerParameter()
        lp.type = "Sigmoid"
        layer = caffe.create_layer(lp)
        bottom = [net.blobs['cls_score']]
        top = [caffe.Blob([])]
        labels = net.blobs['labels'].data
        layer.SetUp(bottom, top)
        layer.Reshape(bottom, top)
        layer.Forward(bottom, top)
        np.set_printoptions(precision=3,suppress=True)
        print("Sigmoid output v.s. Labels: ")
        print(top[0].data)
        print(np.c_[top[0].data, labels])

    def view_softmax_output(self):
        net = self.solver.net
        lp = caffe_pb2.LayerParameter()
        lp.type = "Softmax"
        layer = caffe.create_layer(lp)
        bottom = [net.blobs['cls_score']]
        top = [caffe.Blob([])]
        labels = np.squeeze(net.blobs['labels'].data).astype(np.int64)
        layer.SetUp(bottom, top)
        layer.Reshape(bottom, top)
        layer.Forward(bottom, top)
        np.set_printoptions(precision=3,suppress=True)
        guess_class = np.argmax(top[0].data,axis=1)
        all_class_probs = top[0].data
        class_ave_probs = [] # how confident are the guesses?
        print(all_class_probs.shape)
        for class_index in range(all_class_probs.shape[1]):
            class_probs_indices = np.where(guess_class == class_index)[0]
            class_probs = all_class_probs[class_probs_indices,class_index]
            class_ave_probs.append(np.mean(class_probs))
        if len(class_ave_probs) != self.imdb.num_classes:
            print("Possible error: num_classes for (model,data) = ({},{})".format(len(class_ave_probs),self.imdb.num_classes))
        n_classes = np.max([len(class_ave_probs),self.imdb.num_classes])
        class_ave_probs_np = np.zeros(n_classes)
        class_ave_probs_np[:len(class_ave_probs)] = np.array(class_ave_probs)
        #class_ave_probs = np.mean(all_class_probs,axis=0)
        class_guess_frequency = np.bincount(np.argmax(top[0].data,axis=1),minlength=n_classes)
        class_gt_frequency = np.bincount(labels,minlength=n_classes)
        print("Softmax output v.s. Labels: ")
        print(np.c_[guess_class, labels])
        print(np.c_[class_gt_frequency,class_guess_frequency,class_ave_probs_np])


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    _ = imdb.roidb # initially load the roidb

    if cfg.TRAIN.CLIP_SIZE:
        imdb.resizeRoidbByAnnoSize(cfg.TRAIN.CLIP_SIZE)

    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images() # 
        print('done')

    print('Preparing training data...')
    print(cfg.TASK)
    if cfg.TASK == "object_detection":
        rdl_roidb.prepare_roidb(imdb) # gets image sizes.. might be nice
    elif cfg.TASK == "classification":
        cls_roidb.prepare_roidb(imdb)
    elif cfg.TASK == "regeneration":
        vae_rdl_roidb.prepare_roidb(imdb)

    return imdb.roidb

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.OBJ_DET.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.OBJ_DET.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.OBJ_DET.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,num, num_after))
    return filtered_roidb

def train_net(solver_prototxt, imdb, output_dir,datasetName="",
              pretrained_model=None, solver_state=None, max_iters=400000,
              al_net=None):
    """Train *any* object detection network."""

    #roidb = filter_roidb(roidb)
    print("og solver",solver_prototxt)
    newSolverPrototxt = mangleSolverPrototxt(solver_prototxt,output_dir)
    sw = SolverWrapper(newSolverPrototxt, imdb, output_dir,
                       pretrained_model=pretrained_model,
                       solver_state=solver_state,
                       al_net=al_net)
    print('Solving...')
    model_paths = sw.train_model(max_iters)
    print('done solving')
    return model_paths
