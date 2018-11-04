#!/usr/bin/python
import caffe
import numpy as np
import _init_paths

net = caffe.Net("./models/vae/full/train.prototxt",caffe.TRAIN)


#net.reshape(**{"data": np.arange(100*28*28).reshape(100,28,28)})
#net.forward(**{"data": np.arange(100*28*28).reshape(100,28,28)})

dir(net)
#net.forward(**{"data": np.arange(100*30).reshape(100,30)})

print("HERE")
#print(net)
