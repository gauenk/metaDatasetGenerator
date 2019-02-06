import numpy as np

def setNetworkMasksToOne(net):
    for name,layer in net.layer_dict.items():
        if len(layer.blobs) == 0: continue
        for idx in range(len(layer.blobs)):
            mask = layer.blobs[idx].mask
            layer.blobs[idx].mask[...] = np.ones(mask.shape)

def getLayerInfo(layer):
    if type(layer.bottom) is unicode:
        layer_bottom = str(layer.bottom)
    else:
        layer_bottom = [b for b in layer.bottom]
    if type(layer.top) is unicode:
        layer_top = str(layer.top)
    else:
        layer_top = [b for b in layer.top]
    layer_name = str(layer.name)
    # print(layer_name,layer_top)
    return layer_bottom,layer_top,layer_name
