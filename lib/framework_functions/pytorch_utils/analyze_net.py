import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import _init_paths

from core.pytorch_train import python_object_from_dict
from framework_functions.pytorch_utils.train_net import parse_inputs,get_imdb_dataset
from utils.pytorch_utils import load_train_config,python_function_from_pyfile
from framework_functions.pytorch_utils.adversarial_utils import adversarial_test

import torch
import numpy as np

hook_inputs = {}
hook_outputs = {}

def activations_hook_by_filter_string(layer_index,hook_key,model, input, output):
    global hook_outputs
    global hook_inputs
    print("activation_hook")
    store_input = input[0].cpu().detach().numpy() # the input is a tuple with 1 element
    store_output = output.cpu().detach().numpy()

    if hook_key not in hook_outputs.keys():
        hook_inputs[hook_key] = {}
        hook_outputs[hook_key] = {}
        
    if layer_index in hook_outputs[hook_key].keys():
        hook_inputs[hook_key][layer_index].append(store_input)
        hook_outputs[hook_key][layer_index].append(store_output)
    else:
        hook_inputs[hook_key][layer_index] = [store_input]
        hook_outputs[hook_key][layer_index] = [store_output]

def add_hook_to_model(model,hook_fxn,filter_str):
    from functools import partial
    for index,layer in enumerate(model.children()):
        name = str(layer)
        print(name)
        if filter_str in name:
            print("ADDING THE HOOK")
            # print(dir(layer))
            # print(dir(layer.bias))
            # print(layer.bias.data)
            # layer.bias.data.zero_()
            # print(layer.bias.data.shape)
            # print(layer.weight.data.shape)

            hook_fxn_partial = partial(hook_fxn,index,filter_str)
            layer.register_forward_hook(hook_fxn_partial)
    # count = 0
    # for index,mod in enumerate(model.modules()):
    #     if 'ReLU' not in str(mod):
    #         print("NO")
    #         continue
    #     layer_hook = partial(activations_hook,index)
    #     count += 1
    #     mod.register_forward_hook(layer_hook)
    # print(count)

def data_tensor2mat_for_matmult(data):
    """
    properly handle the "unravel" of the input data
    data is (N_1,M_1); NO (#channels) included.
    data_raveled = data[::-1].ravel()
       ~ and to original shape ~
    data_back = data_raveled.reshape(data.shape)[::-1]
    """
    batch_list = [None for _ in range(data.shape[0])]
    for batch_index in range(data.shape[0]):
        channel_list = [None for _ in range(data.shape[1])]
        for channel_index in range(data.shape[1]):
            elem = data[batch_index][channel_index][::-1].ravel()
            #print("elem.shape: {}".format(elem.shape))
            channel_list[channel_index] = elem
        # across channels 
        channel_elem = np.stack(channel_list,axis=0)[::-1].ravel()
        #print("channel_list.shape: {}".format(channel_elem.shape))
        batch_list[batch_index] = channel_elem
    npmat = np.array(batch_list)
    return npmat


def compute_smoothness(model,device,data_loader):

    input_shape_list = [
        [1, 28, 28],
        [20, 24, 24],
        [20, 17, 17],
        [12, 8, 8]
    ]
    # run over all data
    for imageBlob,scales,sample,index in data_loader.dataset_generator(load_as_blob=True):
        data = imageBlob['data']
        data,target = torch.from_numpy(imageBlob['data']),torch.from_numpy(sample['gt_classes'])
        target = target.long()
        data,target = data.to(device),target.to(device)
        data.requires_grad = True

        output = model(data)

        print(imageBlob['data'].shape)
        mats = data_tensor2mat_for_matmult(imageBlob['data'])
        conv_mats = model_forward_with_mats(model,imageBlob['data'],input_shape_list)
        exit()


        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            continue
        
        print(model)
        print(dir(model))
        print(model.parameters())
        print(model.activations())
        # calculate loss
        loss = criterion(output,target)
        
        # zero gradient
        model.zero_grad()

        # calculate backward
        loss.backward()

        # collect grad data
        data_grad = data.grad.data

        # create an attack
        perturbed_data = fgsm_attack(data,epsilon,data_grad)

        # reclassify sample
        output = model(perturbed_data)

        # check for success
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() != target.item():
            correct += 1
        
    final_acc = correct / float(n_samples)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}"
          .format(epsilon, correct, n_samples, final_acc))
    return final_acc,adv_examples

def load_pytorch_model(model_def,model_state,device):
    print(model_def)
    load_model_func = python_function_from_pyfile(model_def,"get_model").get_model
    model = load_model_func()
    print(model)
    if model_state is not None:
        model.load_state_dict(torch.load(model_state))
        model.eval()
    model.to(device)
    add_hook_to_model(model,activations_hook_by_filter_string,'Conv')
    add_hook_to_model(model,activations_hook_by_filter_string,'Linear')
    return model

def get_linear_function(model,data,batch_size=512):
    data = data.view(batch_size,-1)
    ones = torch.ones(batch_size,1)
    mat_data = torch.cat([data,ones],dim=1)
    layer_mats = []
    for name,param in model.named_parameters():
        if 'conv' in name.lower():
            
            print(param.shape)

def model_forward_with_mats(model,data,input_shape_list):
    global hook_outputs
    global hook_inputs
    from .conv2mat import prepare_pytorch_convolution_parameters_for_toeplitz_construction,format_conv_biases

    input_shape = input_shape_list[0] #data.shape[1:]
    index = 1
    conv_mats = []
    conv_mat_biases = []
    conv_filters = []
    conv_biases = []
    linear_mats = []
    linear_biases = []
    print("[model_forward_with_mats]")
    for name,param in model.named_parameters():
        if 'conv' in name:
            print(name)
            if 'weight' in name:
                # conv_mat = conv_tensor2mat_for_matmult(param,input_shape)
                conv_filters.append(param)
                conv_mat = prepare_pytorch_convolution_parameters_for_toeplitz_construction(param,input_shape)
                input_shape = input_shape_list[index]
                index += 1
                print("name: {}".format(name),conv_mat.shape)
                conv_mats.append(conv_mat)
            if 'bias' in name:
                bias_terms = param.cpu().detach().numpy()
                print(bias_terms.shape)
                conv_biases.append(bias_terms)
                conv_mat_shape = conv_mats[-1].shape # assume the final 'weight' is associated
                bias_input_shape = input_shape_list[index-1]
                bias_term = format_conv_biases(bias_terms,conv_mat_shape,bias_input_shape)
                conv_mat_biases.append(bias_term)
        if 'fc' in name:
            print(name)
            if 'weight' in name:
                mat = param.cpu().detach().numpy()
                print(mat.shape)
                linear_mats.append(mat)
            if 'bias' in name:
                mat = param.cpu().detach().numpy()
                print(mat.shape)
                linear_biases.append(mat)

    numberOfConvLayers = 3
    numberOfLinearLayers = 2
    print("Printing hooks outputs")
    # test the convolution mats:
    convkey_list = {j:i for i,j in enumerate(hook_outputs['Conv'].keys())}
    print(convkey_list)
    linkey_list = {j:i for i,j in enumerate(sorted(hook_outputs['Linear'].keys()))}
    print(linkey_list)

    # gather acitvations for ReLu
    active_masks_data = [None for i in range(numberOfConvLayers)]
    active_masks_bias = [None for i in range(numberOfConvLayers)]
    input_shape_list_index = 0
    
    for key,all_outputs in hook_outputs['Conv'].items():
        outputs_bias = all_outputs[0]
        outputs_data = all_outputs[1]
        print(outputs_bias.shape)
        print(outputs_data.shape)
        mask_bias = (outputs_bias <= 0)
        mask_data = (outputs_data <= 0).ravel()
        print("outputs_data.shape: ",outputs_data.shape)
        print("mask_data.shape: ",mask_data.shape)

        # reorder conv mats&bias
        input_shape = input_shape_list[input_shape_list_index]
        input_shape_list_index += 1

        conv_mat = conv_mats[convkey_list[key]]
        print(conv_mat.shape)
        conv_mats[convkey_list[key]] = reorder_convmat(conv_mat,input_shape,
                                                       outputs_data[0].shape)
        # mask_data = reorder_convmat(mask_data,input_shape,outputs_data[0].shape)
        # mask_data = reorder_vector_by_block(mask_data,outputs_data[0][0][0].shape[0]).astype(np.bool)
        print(mask_data)
        print("mask_data.shape: ",mask_data.shape)
        #reorder_vector_by_block(mask_data,outputs_data[0][0].shape[0])

        conv_bais_term = conv_mat_biases[convkey_list[key]]
        print(conv_bais_term.shape)
        print("all_outputs[1][0][0].shape[0]: ",all_outputs[1][0][0].shape[0])
        print("all_outputs[1][0][0][0].shape[0]: ",all_outputs[1][0][0][0].shape[0])
        conv_mat_biases[convkey_list[key]] = reorder_vector_by_block(conv_bais_term,outputs_data[0][0].shape[0])
        # mask_bias = reorder_vector_by_block(mask_bias,outputs_data[0][0][0].shape[0])
        # reshape for use
        # mask_bias = reorder_matrix_with_channels_by_block(input_data,input_shape)
        # mask_data = reorder_matrix_with_channels_by_block(input_data,input_shape)

        print(mask_data.shape)
        active_masks_data[convkey_list[key]] = mask_data



    # construct activation masks from network outputs
    from .testing_conv2mat import construct_mask,construct_masked_weights,compare_guess_target
    conv_mask = construct_mask(hook_outputs['Conv'],convkey_list,1)
    linear_mask = construct_mask(hook_outputs['Linear'],linkey_list,1)

    # create the input-output matrix
    conv_weights = [conv_mats,conv_mat_biases]
    Wconv,Cconv = construct_masked_weights(numberOfConvLayers,conv_weights,conv_mask)

    linear_weights = [linear_mats,linear_biases]
    Wlin,Clin = construct_masked_weights(numberOfLinearLayers,linear_weights,linear_mask,mask_limit=numberOfLinearLayers-1)

    print(Wconv.shape,Cconv.shape)
    input_data = hook_inputs['Conv'][0][1][0]
    input_shape = input_data.shape
    input_to_mat = input_data.ravel()
    #input_to_mat = reorder_matrix_with_channels_by_block(input_data,input_shape)
    guess = Wconv @ input_to_mat + Cconv
    #guess = reorder_vector_by_block(guess,hook_outputs['Conv'][2][1][0][0][0].shape[0])

    target = hook_outputs['Conv'][4][1].ravel()
    # add the relu
    target[np.where(target <= 0)] = 0
    
    print(guess.shape,target.shape)

    compare_guess_target(guess,target)

    # TEST LINEAR REGIONS
    print("test linear regions")
    input_data = hook_inputs['Linear'][6][1].ravel()
    target = hook_outputs['Linear'][8][1].ravel()

    guess = Wlin @ input_data + Clin
    compare_guess_target(guess,target)

    # test it aaaaaalllllllll
    print("TEST IT ALL!!!")
    input_data = hook_inputs['Conv'][0][1][0]
    input_shape = input_data.shape
    input_to_mat = input_data.ravel()

    W = Wlin @ Wconv
    C = Wlin @ Cconv + Clin

    guess = W @ input_to_mat + C
    
    target = hook_outputs['Linear'][8][1].ravel()
    # add the relu
    
    print(guess.shape,target.shape)

    compare_guess_target(guess,target)

    exit()
    # test this beach #teamLordePerfectPlaces
    inputs = hook_inputs['Conv'][0]
    input_data = inputs[1][0]
    
    target = hook_outputs['Conv'][4]

    
    
    
        

    for key in convkey_list:
        print("key: ",key)
        input = hook_inputs['Conv'][key]
        output = hook_outputs['Conv'][key]

        """
        input[0]: appended bias term
        input[1]: data
        """

        print(input[0].shape)
        print(input[1].shape)
        
        print(output[0].shape)
        print(output[1].shape)

        print(input[0][0][0][0][:10])
        print(input[1][0][0][0][:10])
        print(output[0][0][0][0][:10])
        print(output[1][0][0][0][:10])
        print(output[1][0][0][0][:10] - output[0][0][0][0][:10])
        # print(output[0][0][0][0][:10])
        # print(output[1][0][0][0][:10])

        print(len(output))
        input_bias = input[0][0][0]
        input_data = input[1][0]
        print(input_data.shape)
        print("output[1][0][0][0].shape[0]: ",output[1][0][0][0].shape[0])
        
        if key in [0,2,4]:
            input_shape = input_data.shape
            conv_mat = conv_mats[convkey_list[key]]
            conv_bias = conv_biases[convkey_list[key]]
            conv_filter = conv_filters[convkey_list[key]]
            conv_mat_shape = conv_mat.shape
            bias_term = conv_mat_biases[convkey_list[key]]
            #bias_term = format_conv_biases(conv_bias,conv_mat_shape,input_shape)
            print(bias_term.shape)
            print(conv_mat.shape)
            # input_to_mat = reorder_matrix_with_channels_by_block(bias_term,input_shape)

            print(conv_mat.shape)
            # input_to_mat = reorder_vector_by_block(input_bias.ravel(),input_bias.shape[1])
            # guess_bias = conv_mats[0] @ input_to_mat 
            print(conv_mat[:10])
            # for convolution (not cross-correlation)
            # input_to_mat = reorder_vector_by_block(input_data.ravel(),input_data.shape[1])
            # input_to_mat = reorder_matrix_with_channels_by_block(input_data,input_shape)
            input_to_mat = input_data.ravel()
            # conv_mat = reorder_convmat(conv_mat,input_shape,output[1][0].shape)
            print(input_to_mat.shape)
            guess = conv_mat @ input_to_mat + bias_term
            print(guess.shape)
            # input_to_mat = input_data.ravel()
            # guess = ensure_correctness_of_conv_target(conv_filters[0],input_data)
            # guess = reorder_vector_by_block(guess,32) # convolution comes out tiled (via ensure)

            guess = guess.ravel()
            # start = 4 * 32 + 4 * (32 / 2 - 1) - 16 + 4
            # end = start + 576
            # print("start,end: ",start,end)
            # guess = guess[start:end]
            print(output[1][0][0][0].shape)
            print(output[1][0][0].shape)
            # guess = reorder_vector_by_block(guess,output[1][0][0][0].shape[0])
            print("guess.shape: ",guess.shape)
            print(conv_filters[0][0][0].shape)

            # guess = ensure_correctness_of_pytorch_conv_target(conv_filters[0][0][0],input_data)
            # guess = guess.ravel()

            #guess = reorder_vector_by_block(guess,24)
            # target = output[1][0][0].ravel()
            target = output[1][0].ravel()
            #target = reorder_vector_by_block(target,24)
            print(target.shape)
            # target = target[:576]

            # np.savetxt("guess.csv", guess, delimiter=',', header="guess", comments="")
            # np.savetxt("target.csv", target, delimiter=',', header="target", comments="")

            print(np.c_[guess[:20],target[:20]])
            a = np.c_[guess,target]
            # my guess is the order is incorrect for "isclose"
            np.savetxt("foo.csv", a, delimiter=',', header="A,B", comments="")
            if np.all(np.isclose(guess,target,rtol=1e-03)):
                print("success!")
            else:
                print("keep trying :D")
                indices = np.where(~np.isclose(guess,target))[0]
                # print(np.isclose([guess[indices-1],guess[indices],guess[indices+1]],
                #                  [target[indices-1],target[indices],target[indices+1]]))
                # print(np.isclose([target[indices-1],target[indices],target[indices+1]],
                #                   [guess[indices-1],guess[indices],guess[indices+1]]))
                # print(np.c_[guess[indices-1],target[indices-1]])
                print(np.c_[guess[indices[:10]],target[indices[:10]]])
                # print(np.c_[guess[indices+1],target[indices+1]])
                # print(indices)
                print(len(indices))
                # print(np.dot(conv_mats[0][indices[0]],input_to_mat))

        
    exit()

    return conv_mats
                
    exit()

def reorder_convmat(conv_mat,input_size,output_size):
    nrows,ncols = conv_mat.shape

    n_channels,block_size,block_size = input_size
    block = block_size*block_size
    start = 0
    end = block
    for channel in range(n_channels):
        indices = np.arange(start,end)
        ordered_indices = reorder_vector_by_block(indices,block_size)
        conv_mat[:,start:end] = conv_mat[:,ordered_indices]
        start += block
        end += block

    print(output_size)
    n_channels,block_size,block_size = output_size
    indices = np.arange(block_size*block_size*n_channels)
    ordered_indices = reorder_vector_by_block(indices,block_size)
    conv_mat = conv_mat[ordered_indices,:]

    # block = block_size*block_size
    # start = 0
    # end = block
    # for channel in range(n_channels):
    #     indices = np.arange(start,end)
    #     ordered_indices = reorder_vector_by_block(indices,block_size)
    #     conv_mat[start:end,:] = conv_mat[ordered_indices,:]
    #     start += block
    #     end += block
    return conv_mat

def reorder_matrix_with_channels_by_block(data_matrix,input_size):
    n_channels,block_size,block_size = input_size
    reordered_vector = []
    for channel_index in range(n_channels):
        data_vec = data_matrix[channel_index].ravel()
        print(data_vec.shape)
        print(data_matrix[0].shape)
        newly_ordered = reorder_vector_by_block(data_vec,block_size)
        reordered_vector.extend(newly_ordered)
    reordered_vector = np.array(reordered_vector)
    return reordered_vector

def reorder_vector_by_block(data_vec,block_size):
    # force it to work by adding enough
    # if (len(data_vec) % block_size) != 0:
    #     remainder = block_size - len(data_vec) % block_size
    #     new_data_vec = np.zeros(len(data_vec)+remainder)
    #     new_data_vec[:len(data_vec)] = data_vec
    #     data_vec = new_data_vec

    if (len(data_vec) % block_size) != 0:
        print("[reorder_vector_by_block]: This will not do what you want")
        print(len(data_vec) % block_size)
        exit()
    new_data_vec = np.copy(data_vec)

    start_index = 0
    end_index = block_size
    num_blocks = int(len(data_vec)/block_size)

    data_vec = data_vec[::-1]
    for block_index in range(num_blocks):
        ndv_start = block_index * block_size
        ndv_end = (block_index + 1) * block_size
        new_data_vec[ndv_start:ndv_end] = data_vec[ndv_start:ndv_end][::-1]
        start_index += block_size
        end_index += block_size
    return new_data_vec

def ensure_correctness_of_pytorch_conv_target(pytorch_params,data):
    import torch.nn.functional as F
    import torch.autograd as autograd
    filters = autograd.Variable(pytorch_params.cpu())
    #A test image of a square
    inputs = autograd.Variable(torch.FloatTensor(data))
    filters = filters.reshape(1,1,5,5)
    inputs = inputs.reshape(1,1,28,28)
    print(filters.shape)
    print(inputs.shape)
    results = F.conv2d(inputs, filters) 
    return results.detach().numpy()

def ensure_correctness_of_conv_target(pytorch_params,data):
    filter_mat_list = pytorch_params.cpu().detach().numpy()
    agg_output = []
    from scipy.signal import convolve2d
    print(data.shape)
    count = 0
    for filter_mat_with_channel in filter_mat_list:
        # assume single channel
        filter_mat = filter_mat_with_channel[0]
        filter_mat = filter_mat[::-1]
        filter_mat = filter_mat[:,::-1]
        print(filter_mat.shape)
        output = convolve2d(data,filter_mat)
        agg_output.append(output)
        if count >= 0:
            break
        count += 1
    agg_output = np.array(agg_output).ravel()
    print("[ensure_correctness_of_conv_target]: agg_output")
    print(agg_output.shape)
    return agg_output

    
def forward_dummy_data(model,device):
    sdata = torch.ones(1,1,28,28)
    sdata = sdata.to(device)
    model(sdata)

def analyze_net_pytorch(**kwargs):
    args = parse_inputs(**kwargs)
    print('Called with args:')
    print(args)
    train_config = load_train_config(args.cfg_file,args)
    imdb,roidb = get_imdb_dataset(args)
    data_loader = imdb.create_data_loader(train_config,None,None)
    #data_loader = get_data_loader(imdb,train_config,al_net=None)
    device = torch.device('cuda:0')    
    model = load_pytorch_model(args.model_def,args.model_state,device)

    forward_dummy_data(model,device)

    parameters = model.parameters()
    criterion = python_object_from_dict(train_config.model_info.criterion_info,"pytorch_code",parameters)
    epsilon_list = [0.15,0.1,0.05,0.035,0.025,0.015,0.001,0.0001,0.0]
    # adversarial_test(model, device, data_loader, criterion, epsilon_list)
    compute_smoothness(model,device,data_loader)
    print("BYE KENT")
    exit()


    
