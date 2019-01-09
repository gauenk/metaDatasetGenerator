import matplotlib,glob,cv2
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from fresh_config import cfg,create_model_name,update_one_current_config_field,reset_model_name
from fresh_util import transpose_measures,assert_equal_lists,get_unique_experiment_field_change

def plot_with_yerror_bars_template_function(xdata,ydata,yerror,line_type,xlabel,ylabel,title):
    hdle, = plt.errorbars(xdata,ydata,line_type,yerr=yerror)
    if title: plt.title(title)
    if ylabel: plt.ylabel(ylabel)
    if xlabel: plt.xlabel(xlabel)
    return hdle

def plot_template_function(xdata,ydata,line_type,xlabel,ylabel,title):
    hdle, = plt.plot(xdata,ydata,line_type)
    if title: plt.title(title)
    if ylabel: plt.ylabel(ylabel)
    if xlabel: plt.xlabel(xlabel)
    return hdle

def plot_add_legend(plot_handles,plot_handle_names):
    plt.legend(plot_handles, plot_handle_names)
    
def plot_clear():
    plt.clf(),plt.cla()

def plot_save_figure(filename):
    plt.savefig(filename)
    
def plot_measure_vs_clusters(train_scores,test_scores,kmeans_search_list,ylabel,title,filename):
    ave_scores = average_list_elements(train_scores,test_scores)
    train_handle = plot_template_function(kmeans_search_list,train_scores,'b+',
                                          '# of clusters',ylabel,title)
    test_handle = plot_template_function(kmeans_search_list,test_scores,'g^',
                                         None,None,None)
    ave_handle = plot_template_function(kmeans_search_list,ave_scores,'r*',
                                         None,None,None)
    plot_add_legend([train_handle,test_handle,ave_handle],['train','test','ave'])
    plot_save_figure(filename)
    plot_clear()

def plot_aggregate_scores_vs_clusters(train_scores,test_scores,kmeans_search_list):
    plot_measure_vs_clusters(train_scores,test_scores,kmeans_search_list,'Combined [via sum] Score [Measure]',
                             'Choosing Optimal K [{}]'.format(cfg.modelInfo.name),
                             'measure_combined_vs_cluster_model_{}.png'.format(cfg.modelInfo.name))

def plot_measure(train_measure,test_measure,measure_name,kmeans_search_list):
    plot_measure_vs_clusters(train_measure,test_measure,kmeans_search_list,'{} [Measure] '.format(measure_name),
                             'Choosing Optimal K [{}]'.format(cfg.modelInfo.name),
                             'measure_{}_vs_cluster_model_{}.png'.format(measure_name,cfg.modelInfo.name))

def plot_measure_list(train_measure_list,test_measure_list,kmeans_search_list):
    """
    measure_list:
       -> len(measure_list) = # of k's in search list
       -> len(measure_list[index]) = # of measures
    measure_list_transpose:
       -> len(measure_list) = # of measures
       -> len(measure_list[index]) = # of k's in search list
    """
    measure_name_dict = {'cluster':['silhouette','homogeneity_ds_labels','homogeneity_correct'],
                         'separability':['separability_ds_labels','separability_ds_correct']}
    for measure_type in train_measure_list.keys():
        measure_name_list = measure_name_dict[measure_type]
        train_measure_type_list_t,measure_name_list_t = transpose_measures(train_measure_list[measure_type],measure_name_list)
        test_measure_type_list_t,measure_name_list_t = transpose_measures(test_measure_list[measure_type],measure_name_list)
        for train_measure,test_measure,measure_name in zip(train_measure_type_list_t,test_measure_type_list_t,measure_name_list_t):
            print(measure_name)
            plot_measure(train_measure,test_measure,measure_name,kmeans_search_list)


def init_plot(plot_dict,fieldname,msr_name):
    title = fieldname
    # title = fieldname + '_' + msr_name
    plot_dict[msr_name] = plt.subplots()
    plot_dict[msr_name][1].set_title(title)
    plot_dict[msr_name][1].set_xlabel('number of clusters')
    plot_dict[msr_name][1].set_ylabel(msr_name)

def add_plot_data(plot_dict,msr_name,msr_value,fieldname,fieldvalue,setid,marker,kmeans_search_list):
    # label = setid + '_' + fieldname + '_' + str(fieldvalue)
    label = str(fieldvalue)
    if setid == 'train': label = '_'+label
    if setid == 'train': fmt = 'b'+marker+'-'
    elif setid == 'test': fmt = 'g'+marker+'-'
    plot_dict[msr_name][1].plot(kmeans_search_list,msr_value,fmt,label=label)

def add_plot_with_yerror_data(plot_dict,msr_name,msr_value,msr_error,fieldname,fieldvalue,setid,marker,kmeans_search_list):
    if fieldvalue is not None: label = str(fieldvalue)
    else: label = '_'
    if setid == 'train': label = '_'+label
    if setid == 'train': fmt = 'b'+marker+'-'
    elif setid == 'test': fmt = 'g'+marker+'-'
    plot_dict[msr_name][1].errorbar(kmeans_search_list,msr_value,yerr=msr_error,fmt=fmt,label=label)


def average_list_elements(list_a,list_b):
    list_c = [(a+b)/2. for a,b in zip(list_a,list_b)]
    return list_c
        

def tile_by_comboid(combolist):
    xsep = 10 # number of pixels between images along x
    ysep = 10 # number of pixels between images along y
    
    for comboid in combolist:
        imglist = []
        for filename in glob.glob("measure*+{}_*png".format(comboid)):
            if 'tile' in filename: continue # don't add to itself
            imglist.append(cv2.imread(filename))
        yshapes = [image.shape[0] for image in imglist]
        xshapes = [image.shape[1] for image in imglist]
        shape = [max(yshapes),sum(xshapes)+xsep*len(xshapes),3]
        tile = np.zeros(shape)
        x_index_start,x_index_end = 0,0
        for index,image in enumerate(imglist):
            if index != 0: x_index_start += xshapes[index-1] + xsep
            x_index_end = x_index_start + xshapes[index]
            tile[:image.shape[0],x_index_start:x_index_end,:] = image
        cv2.imwrite("tile_{}_{}.png".format(comboid,cfg.modelInfo.name),tile)
                                                  

def mangle_plot_name(modelInfo,fieldname):
    import copy
    modelInfo_copy = copy.deepcopy(modelInfo)
    modelInfo_copy[fieldname] = '[{}]'.format(fieldname)
    return create_model_name(modelInfo_copy)
    
def analyze_expfield_results(train_results,test_results,exp_configs,expfield_filter,kmeans_search_list):
    measure_name_dict = cfg.measure_name_dict
    marker_list = cfg.plot.marker_list
    unique_changes = get_unique_experiment_field_change(exp_configs)
    marker_index = dict.fromkeys(unique_changes,0)
    plot_dict = {}
    special_plot_dict = {}
    for field_index,exp_config in enumerate(exp_configs):
        # update to exp_config
        fieldname,fieldvalue = update_one_current_config_field(exp_config)
        if fieldname != expfield_filter: continue
        reset_model_name()

        # update to exp_config
        train_msrs,test_msrs = train_results[field_index],test_results[field_index]
        print(fieldname)
        print(train_msrs)
        for measure_type in train_msrs.keys():
            msr_dict_type = measure_name_dict[measure_type]
            tr_type_t,tr_msr_name_t = transpose_measures(train_msrs[measure_type],msr_dict_type)
            te_type_t,te_msr_name_t = transpose_measures(test_msrs[measure_type],msr_dict_type)
            assert_equal_lists(tr_msr_name_t,te_msr_name_t)
            for tr_msr,te_msr,msr_name in zip(tr_type_t,te_type_t,tr_msr_name_t):
                plot_title = mangle_plot_name(cfg.modelInfo,fieldname)
                marker = marker_list[marker_index[fieldname]]
                special_plot_init(special_plot_dict,msr_name,plot_title,fieldname)
                add_special_plot_data(special_plot_dict,msr_name,tr_msr,te_msr,marker,kmeans_search_list,fieldname,fieldvalue)
                if msr_name not in plot_dict.keys(): init_plot(plot_dict,plot_title,msr_name)
                add_plot_data(plot_dict,msr_name,tr_msr,fieldname,fieldvalue,'train',marker,kmeans_search_list)
                add_plot_data(plot_dict,msr_name,te_msr,fieldname,fieldvalue,'test',marker,kmeans_search_list)
        marker_index[fieldname] += 1

    for exp_change in unique_changes:
        for measure_type in train_msrs.keys():
            msr_dict_type = measure_name_dict[measure_type]
            _,tr_msr_name_t = transpose_measures(train_msrs[measure_type],msr_dict_type)
            for msr_name in tr_msr_name_t:
                if msr_name not in plot_dict.keys(): init_plot(plot_dict,msr_name) #error
                save_plot_data(plot_dict,msr_name,exp_change)
                save_special_plot_data(special_plot_dict,msr_name,exp_change)
                # save_name = msr_name+'_'+ exp_change + '.png'
                # adjust_plot_limits(plot_dict[msr_name][1],.05,.01)
                # plot_handles,plot_labels = add_train_test_patch_handles(plot_dict[msr_name][1])
                # add_legend_to_axis(plot_dict[msr_name],plot_handles,plot_labels,exp_change)
                # plot_dict[msr_name][0].savefig(save_name,bbox_inches='tight')

def save_plot_data(plot_dict,msr_name,exp_change):
    save_name = msr_name+'_'+ exp_change + '.png'
    adjust_plot_limits(plot_dict[msr_name][1],.05,.01)
    plot_handles,plot_labels = add_train_test_patch_handles(plot_dict[msr_name][1])
    add_legend_to_axis(plot_dict[msr_name],plot_handles,plot_labels,exp_change)
    plot_dict[msr_name][0].savefig(save_name,bbox_inches='tight')
    
def add_legend_to_axis(plot_list,lgd_handles,lgd_names,lgd_title):
    plot_list[0].subplots_adjust(right=0.7)
    plot_list[0].tight_layout(rect=[0,0,0.75,1])
    if lgd_handles is None or lgd_names is None:
        plot_list[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title=lgd_title)
    else:
        plot_list[1].legend(handles=lgd_handles,labels=lgd_names,loc='center left', bbox_to_anchor=(1, 0.5), title=lgd_title)

def adjust_plot_limits(axis_var,xinc,yinc):
    ylim = axis_var.get_ylim()
    xlim = axis_var.get_xlim()
    ylim = [ylim[0]-yinc*ylim[1],(1.+yinc)*ylim[1]]
    xlim = [xlim[0]-xinc*xlim[1],(1.+xinc)*xlim[1]]
    axis_var.set_ylim(ylim)
    axis_var.set_xlim(xlim)
    
def add_train_test_patch_handles(axis_var):
    train_patch = mpatches.Patch(color=cfg.plot.color_dict['train'],label='train')
    test_patch = mpatches.Patch(color=cfg.plot.color_dict['test'],label='test')
    plot_handles = [train_patch,test_patch]
    plot_labels = ['train','test']
    handles,labels = axis_var.get_legend_handles_labels()
    plot_handles += handles
    plot_labels += labels
    return plot_handles,plot_labels


def is_msr_name_in_special_plot_dict(special_plot_dict,msr_name):
    # is_a_special_plot,is_found_in_dict
    if msr_name not in cfg.special_plot_name_dict.keys(): return False,None
    special_names = cfg.special_plot_name_dict[msr_name]    
    dict_keys = special_plot_dict.keys()
    rtn_list = {}
    for special_name in special_names:
        rtn_list[special_name] = False
        if special_name in dict_keys: rtn_list[special_name] = True
    return True,rtn_list
    
def special_plot_init(special_plot_dict,msr_name,plot_title,fieldname):
    isIn,isFound = is_msr_name_in_special_plot_dict(special_plot_dict,msr_name)
    if isIn is False: return
    # not isFound = True ==> isIn = True
    special_names = cfg.special_plot_name_dict[msr_name]
    for special_name in special_names:
        if isFound[special_name]: continue
        special_init_plot_func = cfg.special_plot_func_dict[special_name][0]
        special_init_plot_func(special_plot_dict,plot_title,special_name,fieldname)

def add_special_plot_data(special_plot_dict,msr_name,tr_msr,te_msr,marker,kmeans_search_list,fieldname,fieldvalue):
    isIn,isFound = is_msr_name_in_special_plot_dict(special_plot_dict,msr_name)    
    if isIn is False: return
    special_names = cfg.special_plot_name_dict[msr_name]
    for special_name in special_names:
        if isFound[special_name] is False: continue
        special_plot_function = cfg.special_plot_func_dict[special_name][1]
        special_plot_function(special_plot_dict,special_name,tr_msr,te_msr,marker,kmeans_search_list,fieldname,fieldvalue)

def save_special_plot_data(special_plot_dict,msr_name,exp_change):
    isIn,isFound = is_msr_name_in_special_plot_dict(special_plot_dict,msr_name)    
    if isIn is False: return
    special_names = cfg.special_plot_name_dict[msr_name]
    for special_name in special_names:
        if isFound[special_name] is False: continue
        special_plot_function = cfg.special_plot_func_dict[special_name][2]
        special_plot_function(special_plot_dict,special_name,exp_change)

def init_data_set_error_with_generalization_error_bars(special_plot_dict,plot_title,special_name,fieldname):
    title = 'error_with_generaliation_error_bars'
    special_plot_dict[special_name] = plt.subplots()
    special_plot_dict[special_name][1].set_title(title)
    special_plot_dict[special_name][1].set_xlabel(fieldname) #iterations
    special_plot_dict[special_name][1].set_ylabel(special_name)

def add_data_set_error_with_generalization_error_bars(special_plot_dict,special_name,tr_msr,te_msr,marker,kmeans_search_list,fieldname,fieldvalue,setID):
    difference = [tr_msr[0] - te_msr[0]]
    if setID == 'train': y_data = [tr_msr[0]]
    elif setID == 'test': y_data = [te_msr[0]]
    else: raise ValueError("unknown setid: {}".format(setID))
    x_data = [fieldvalue]
    add_plot_with_yerror_data(special_plot_dict,special_name,y_data,difference,None,None,setID,marker,x_data)

def save_data_set_error_with_generalization_error_bars(special_plot_dict,special_name,exp_change,setID):
    save_name = special_name+'_'+ exp_change +'_'+setID+'_'+'error_with_generalization_error.png'
    adjust_plot_limits(special_plot_dict[special_name][1],.05,.01)    
    # plot_handles,plot_labels = axis_var.get_legend_handles_labels()
    #add_legend_to_axis(special_plot_dict[special_name],plot_handles,plot_labels,exp_change)
    special_plot_dict[special_name][0].savefig(save_name,bbox_inches='tight')    
    
def add_data_train_error_with_generalization_error_bars(special_plot_dict,special_name,tr_msr,te_msr,marker,kmeans_search_list,fieldname,fieldvalue):
    add_data_set_error_with_generalization_error_bars(special_plot_dict,special_name,tr_msr,te_msr,marker,kmeans_search_list,fieldname,fieldvalue,'train')

def add_data_test_error_with_generalization_error_bars(special_plot_dict,special_name,tr_msr,te_msr,marker,kmeans_search_list,fieldname,fieldvalue):
    add_data_set_error_with_generalization_error_bars(special_plot_dict,special_name,tr_msr,te_msr,marker,kmeans_search_list,fieldname,fieldvalue,'test')


def save_data_train_error_with_generalization_error_bars(special_plot_dict,special_name,exp_change):
    save_data_set_error_with_generalization_error_bars(special_plot_dict,special_name,exp_change,'train')

def save_data_test_error_with_generalization_error_bars(special_plot_dict,special_name,exp_change):
    save_data_set_error_with_generalization_error_bars(special_plot_dict,special_name,exp_change,'test')
    
# set before running
cfg.special_plot_func_dict = {'train_with_gen_error':[init_data_set_error_with_generalization_error_bars,add_data_train_error_with_generalization_error_bars,save_data_train_error_with_generalization_error_bars],'test_with_gen_error':[init_data_set_error_with_generalization_error_bars,add_data_test_error_with_generalization_error_bars,save_data_test_error_with_generalization_error_bars]}
