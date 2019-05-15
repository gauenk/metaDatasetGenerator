import numpy as np

def test_conv2mat_at_layer_index(layer_index,network_outputs_by_layer,conv_mats_by_layer,data):
    network_output = network_outputs_by_layer[layer_index]
    conv_mat = conv_mats[layer_index]
    formatted_sample = reorder_matrix_with_channels_by_block(data,data.shape)
    

def compare_guess_target(guess,target):
    print(np.c_[guess[:20],target[:20]])
    a = np.c_[guess,target]
    np.savetxt("foo.csv", a, delimiter=',', header="A,B", comments="")
    if np.all(np.isclose(guess,target,rtol=1e-3)):
        print("success")
    else:
        print("keep trying :D")
        indices = np.where(~np.isclose(guess,target))[0]
        print(len(indices))
        print(np.c_[guess[indices[:10]],target[indices[:10]]])


def construct_masked_weights(numberOfLayers,weights,mask,mask_limit=None):
    if mask_limit is None:
        mask_limit = numberOfLayers
    W,C = None,None
    mats,biases = weights[0],weights[1]
    for index in range(0,numberOfLayers):
        mask_indices = np.where(mask[index])[0]
        if index < mask_limit:
            mats[index][mask_indices,:] = 0
            biases[index][mask_indices] = 0
        if index == 0:
            W = mats[index]
            C = biases[index]
        else:
            W = mats[index] @ W
            C = mats[index] @ C + biases[index]
    return W,C


def construct_mask(output_dict,key_list,index):
    mask = [None for _ in key_list]
    for key,all_outputs in output_dict.items():
        data = all_outputs[index].squeeze()
        mask_data = (data <= 0).ravel()
        mask[key_list[key]] = mask_data
    return mask


