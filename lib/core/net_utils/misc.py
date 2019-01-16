import numpy as np

def setNetworkMasksToOne(net):
    for name,layer in net.layer_dict.items():
        if len(layer.blobs) == 0: continue
        for idx in range(len(layer.blobs)):
            mask = layer.blobs[idx].mask
            layer.blobs[idx].mask[...] = np.ones(mask.shape)

