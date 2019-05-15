import torch
import numpy as np

def format_conv_biases(conv_biases,conv_mat_shape,input_shape):
    # conv_biases = pytorch_conv_biases.cpu().detach().numpy()
    n_filters = len(conv_biases)
    n_input_channels = input_shape[0]
    input_mat_shape = input_shape[1:]
    C = []
    count = 0
    nrows = int(conv_mat_shape[0]/n_filters)
    for filter_index in range(n_filters):
        # bias_by_channel = []
        conv_bias_by_filter = conv_biases[filter_index]
        # this gets the output size
        # there is not channel
        #conv_bias_by_filter_channel = conv_bias_by_filter[channel_index]
        print(conv_bias_by_filter)
        bias_tile = np.repeat(conv_bias_by_filter,nrows)
        # if len(bias_by_channel) > 0:
        #     bias_by_channel = np.c_[bias_by_channel,bias_tile]
        # else:
        #     bias_by_channel = bias_tile
        # bias_by_channel = np.array(bias_by_channel)
        # print("bias_by_channel.shape",bias_by_channel.shape)
        if len(C) > 0:
            C = np.r_[bias_tile,C]
        else:
            C = bias_tile
        # # sub-testing
        # if count >= 0:
        #     break
        count += 1
    return C

def compute_output_shape_from_single_filter(filter_mat,input_shape):
    """
    assume all padding needed is given
    assume stride of correlation is 1
    
    *warning: this function is only used for testing the correctness of the other functions
    """
    input_bs,input_channels,input_rows,input_cols = input_shape
    filter_channels, filter_rows, filter_cols = filter_mat.shape
    output_rows = input_rows + filter_rows - 1 
    output_cols = input_cols + filter_cols - 1
    output_shape = (output_rows,output_cols)
    return output_shape

def compute_toeplitz_matrix_shape_from_single_filter(filter_mat,input_shape,output_shape = None):
    """
    assume all padding needed is given
    assume stride of correlation is 1
    """
    if output_shape is None:
        output_shape = compute_output_shape_from_single_filter(filter_mat,input_shape)
    input_bs,input_channels,input_rows,input_cols = input_shape
    filter_channels, filter_rows, filter_cols = filter_mat.shape
    toeplitz_rows = input_cols
    toeplitz_cols = filter_rows * filter_cols
    toeplitz_shape = (toeplitz_rows,toeplitz_cols)
    # TODO: finish this shaping!
    return toeplitz_shape

def construct_doubly_toeplitz_matrix_from_single_filter(filter_mat,input_shape,output_shape = None):
    """
    filter: a (f1,f2) size filter applied to the input
    input_shape: (#channels,length,width)
    """
    if output_shape is None:
        output_shape = compute_output_shape_from_single_filter(filter_mat,input_shape)        
    toeplitz_shape = compute_toeplitz_matrix_shape_from_single_filter(filter_mat,input_shape,output_shape)

    # put filter in bottom left corner
    top_pad = (output_cols - filter_cols,0)
    right_pad = (0,output_rows - filter_rows)
    zero_padded_filter = np.pad(filter_mat,(top_pad,right_pad),'constant',constant_values=0)
    
    # create the doubly toeplitz matrix from the single zero padded filter
    w = construct_doubly_toeplitz_matrix_from_single_zero_padded_filter(zero_padded_filter,input_nrows,input_ncols)
    return w
    
def construct_toeplitz_matrix_for_set_of_filters(filter_mat_list,input_shape):
    """
    INPUT:

    filter_set: a list of filters [ mat_1, mat_2, ..., mat_{# of filters}]
    input_shape: (batch_size,#channels,length,width)
    
    OUTPUT:
    
    W = [ w_1; w_2; ...; w_n] with n = # filters
        the input shape should include the width and height of each vector

    """
    n_input_channels = input_shape[0]
    input_mat_shape = input_shape[1:]
    # print(input_mat_shape)
    # print(filter_mat_list.shape)
    W = []
    count = 0
    for filter_mat in filter_mat_list:
        w = []
        for channel_index in range(n_input_channels):
            # flip the filter before it goes in
            # since this function is actually implemented for convolution;
            # _not_ cross-correlation
            filter_mat_flipped = filter_mat[channel_index]
            filter_mat_flipped = filter_mat_flipped[::-1]
            filter_mat_flipped = filter_mat_flipped[:,::-1]
            w_filter = mat_from_convfilter(filter_mat_flipped,input_mat_shape)
            if len(w) > 0:
                w = np.c_[w,w_filter]
            else:
                w = w_filter
        w = np.array(w)
        # print("w.shape: ",w.shape)
        if len(W) > 0:
            W = np.r_[w,W]
        else:
            W = w 
        # # sub-testing
        # if count >= 0:
        #     break
        count += 1
    W = np.array(W)
    return W

def prepare_pytorch_convolution_parameters_for_toeplitz_construction(parameters,input_shape):
    # I don't think we actually have to do anything.
    filter_mat_list = parameters.cpu().detach().numpy()
    matrix = construct_toeplitz_matrix_for_set_of_filters(filter_mat_list,input_shape)    
    return matrix
    

def conv_tensor2mat_for_matmult(params_t,input_shape):
    params = params_t.cpu().detach().numpy()
    # params.shape : ( num output channels, num input channels, filter size 1 , filter size 2)
    # data.shape : ( batch size, channels, width, height)
    # print("params.shape",params.shape)
    # input_channel = data[0].reshape(input_shape[1],input_shape[2]*input_shape[3])
    # print("input_shape: {}".format(input_shape))
    bs,nchannels,nrows,ncols = input_shape
    assert len(params.shape) == 4, "we must have dim 4 for weights"
    output_channel_list = []
    for output_channel_index in range(params.shape[0]):
        input_channel_list = []
        for input_channel_index in range(params.shape[1]):
            convfilter = params[output_channel_index][input_channel_index]
            param_mat = mat_from_convfilter(convfilter,input_shape)
            # print("param_mat.shape",param_mat.shape)
            input_channel_list.append(param_mat)
        # do the same funtion on *this* 2d "conv filter":
        #    ~ applied across channels instead of an image ~ 
        channel_params = np.stack(input_channel_list,axis=0)
        output_channel_list.append(channel_params)
        #channel_mat = mat_from_convfilter(channel_params,input_channel)
    conv_mat = np.stack(output_channel_list,axis=0)
    # print("conv_mat.shape",conv_mat.shape)
    return conv_mat

def mat_from_convfilter(filter_mat,input_shape):
    input_nrows,input_ncols = input_shape
    filter_rows,filter_cols = filter_mat.shape
    padded_convfilter = conv2padded_for_matmult(input_shape,filter_mat)
    # print("padded_convfilter.shape",padded_convfilter.shape)
    toep_convfilter = construct_doubly_toeplitz_matrix_from_single_zero_padded_filter(padded_convfilter,input_nrows,input_ncols,filter_rows,filter_cols)
    # print("toep_convfilter.shape",toep_convfilter.shape)
    return toep_convfilter

def conv2padded_for_matmult(input_shape,filter_mat):
    input_rows, input_cols = input_shape
    filter_rows, filter_cols = filter_mat.shape
    # online says: input_rows + filter_rows - 1
    # possibly the actual computation is "input_rows - filter_rows + 1"
    num_boundary_not_included_rows = 0 # 2*(filter_rows - 1)
    num_boundary_not_included_cols = 0 # 2*(filter_cols - 1)
    output_rows = input_rows + filter_rows - 1 - num_boundary_not_included_rows
    output_cols = input_cols + filter_cols - 1 - num_boundary_not_included_cols
    # print(filter_rows,input_rows)
    # print(filter_cols,input_cols)
    # print(output_rows)
    # print(output_cols)
    # print(filter_mat.shape)
    # print("output dim: ({},{})".format(output_rows,output_cols))
    top_pad = (output_rows - filter_rows,0)
    right_pad = (0,output_cols - filter_cols)
    if np.any([output_rows - filter_rows < 0, output_cols - filter_cols < 0]):
        zero_padded_filter = filter_mat
    else:
        zero_padded_filter = np.pad(filter_mat,(top_pad,right_pad),'constant',constant_values=0)
    # print("zero_padded_filter.shape: ",zero_padded_filter.shape)
    return zero_padded_filter

def construct_doubly_toeplitz_matrix_from_single_zero_padded_filter(padded_filter_mat,input_nrows,input_ncols,filter_rows,filter_cols):
    from scipy.linalg import toeplitz

    output_rows,output_cols = padded_filter_mat.shape
    output_indices = np.arange(output_rows*output_cols).reshape(output_rows,output_cols)
    # the next line of code says: no padding; stride = 1;
    output_indices = output_indices[(filter_rows-1):-(filter_rows-1),
                                    (filter_cols-1):-(filter_cols-1)]

    # init toeplitz list
    toeplitz_list = []
    for index,row in enumerate(padded_filter_mat[::-1]):
        c = row
        r = np.r_[row[0],np.zeros(input_ncols - 1)]
        toep = toeplitz(c,r)
        toeplitz_list.append(toep)
    # print(len(toeplitz_list[0]))
    # print(len(toeplitz_list))
    # exit()
    
    # doubly padded toeplitz (indices)
    c = range(1, padded_filter_mat.shape[0]+1)
    r = np.r_[c[0], np.zeros(input_nrows-1, dtype=int)]
    doubly_indices = toeplitz(c, r)
    # print(doubly_indices.shape)

    # doubly blocked matrix (zero values)
    toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    # print(doubly_blocked_shape)
    # tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape # hight and withs of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            row_index = i
            col_index = j
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i:end_i, start_j:end_j] = toeplitz_list[doubly_indices[row_index,col_index]-1]

    doubly_blocked = doubly_blocked[output_indices.ravel(),:]
    return doubly_blocked
